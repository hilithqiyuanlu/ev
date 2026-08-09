import Foundation

final class EngineClient: @unchecked Sendable {
    var onEvent: (@Sendable (EngineEnvelope) -> Void)?
    var onTermination: (@Sendable (Int32) -> Void)?
    var onStderr: (@Sendable (String) -> Void)?

    private let process = Process()
    private let inputPipe = Pipe()
    private let outputPipe = Pipe()
    private let errorPipe = Pipe()
    private let queue = DispatchQueue(label: "ev.engine.client")
    private var outputBuffer = Data()

    var isRunning: Bool { process.isRunning }

    func start() throws {
        guard !process.isRunning else { return }
        let repoRoot = Self.repositoryRoot
        let python = repoRoot.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            throw EngineClientError.pythonMissing(python.path)
        }
        let support = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appendingPathComponent("EV", isDirectory: true)
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)

        process.executableURL = python
        process.arguments = ["-m", "ev", "engine", "serve"]
        process.currentDirectoryURL = repoRoot
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["EV_DATA_DIR"] = support.path
        environment["EV_MODEL_ROOT"] = support.appendingPathComponent("models").path
        process.environment = environment
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        outputPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.consume(handle.availableData)
        }
        errorPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let message = String(data: data, encoding: .utf8) else { return }
            self?.onStderr?(message)
        }
        process.terminationHandler = { [weak self] process in
            self?.onTermination?(process.terminationStatus)
        }
        try process.run()
    }

    func send(_ command: String, payload: [String: Any] = [:]) {
        let request: [String: Any] = [
            "version": 1,
            "request_id": UUID().uuidString,
            "command": command,
            "payload": payload,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: request) else { return }
        queue.async { [weak self] in
            guard let self, self.process.isRunning else { return }
            guard var line = String(data: data, encoding: .utf8) else { return }
            line.append("\n")
            self.inputPipe.fileHandleForWriting.write(Data(line.utf8))
        }
    }

    func shutdown() {
        guard process.isRunning else { return }
        send("shutdown")
        queue.asyncAfter(deadline: .now() + 3) { [weak self] in
            guard let self, self.process.isRunning else { return }
            self.process.terminate()
        }
    }

    private func consume(_ data: Data) {
        guard !data.isEmpty else { return }
        queue.async { [weak self] in
            guard let self else { return }
            self.outputBuffer.append(data)
            while let newline = self.outputBuffer.firstIndex(of: 0x0A) {
                let line = self.outputBuffer[..<newline]
                self.outputBuffer.removeSubrange(...newline)
                guard !line.isEmpty,
                      let event = try? JSONDecoder().decode(EngineEnvelope.self, from: Data(line)) else { continue }
                self.onEvent?(event)
            }
        }
    }

    static var repositoryRoot: URL {
        if let configured = ProcessInfo.processInfo.environment["EV_REPO_ROOT"] {
            return URL(fileURLWithPath: configured, isDirectory: true)
        }
        var source = URL(fileURLWithPath: #filePath)
        // EngineClient.swift -> Services -> EV -> macos -> apps -> repository root.
        for _ in 0..<5 { source.deleteLastPathComponent() }
        return source
    }
}

enum EngineClientError: LocalizedError {
    case pythonMissing(String)

    var errorDescription: String? {
        switch self {
        case .pythonMissing(let path): "未找到开发环境 Python：\(path)"
        }
    }
}

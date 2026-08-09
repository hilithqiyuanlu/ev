import Darwin
import Foundation

protocol EngineTransport: AnyObject {
    var onEvent: (@Sendable (EngineEnvelope) -> Void)? { get set }
    var onTermination: (@Sendable (Int32) -> Void)? { get set }
    var onStderr: (@Sendable (String) -> Void)? { get set }
    var isRunning: Bool { get }
    func start() throws
    func send(_ command: String, payload: [String: Any])
    func shutdown()
}

extension EngineTransport {
    func send(_ command: String) { send(command, payload: [:]) }
}

final class EngineClient: EngineTransport, @unchecked Sendable {
    var onEvent: (@Sendable (EngineEnvelope) -> Void)?
    var onTermination: (@Sendable (Int32) -> Void)?
    var onStderr: (@Sendable (String) -> Void)?

    private let process = Process()
    private let inputPipe = Pipe()
    private let outputPipe = Pipe()
    private let errorPipe = Pipe()
    private let queue = DispatchQueue(label: "ev.engine.client")
    private let outputReadQueue = DispatchQueue(label: "ev.engine.stdout")
    private let errorReadQueue = DispatchQueue(label: "ev.engine.stderr")
    private let inputLock = NSLock()
    private let repositoryRootOverride: URL?
    private let supportDirectoryOverride: URL?
    private var outputBuffer = Data()

    var isRunning: Bool { process.isRunning }

    init(repositoryRoot: URL? = nil, supportDirectory: URL? = nil) {
        self.repositoryRootOverride = repositoryRoot
        self.supportDirectoryOverride = supportDirectory
    }

    deinit {
        onStderr?("Engine client released\n")
    }

    func start() throws {
        guard !process.isRunning else { return }
        let repoRoot = repositoryRootOverride ?? Self.repositoryRoot
        let python = repoRoot.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            throw EngineClientError.pythonMissing(python.path)
        }
        let support = try supportDirectoryOverride ?? FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appendingPathComponent("EV", isDirectory: true)
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        let logs = support.appendingPathComponent("logs", isDirectory: true)
        try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let logPath = logs.appendingPathComponent("ev.log")

        process.executableURL = python
        process.arguments = ["-m", "ev", "--log-path", logPath.path, "engine", "serve"]
        process.currentDirectoryURL = repoRoot
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["EV_DATA_DIR"] = support.path
        environment["EV_MODEL_ROOT"] = support.appendingPathComponent("models").path
        process.environment = environment
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        process.terminationHandler = { [weak self] process in
            self?.onTermination?(process.terminationStatus)
        }
        try process.run()
        startReaders()
        onStderr?("Engine PID \(process.processIdentifier) · \(python.path)\n")
    }

    func send(_ command: String, payload: [String: Any] = [:]) {
        let request: [String: Any] = [
            "version": 1,
            "request_id": UUID().uuidString,
            "command": command,
            "payload": payload,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: request) else { return }
        guard var line = String(data: data, encoding: .utf8) else { return }
        line.append("\n")
        inputLock.lock()
        defer { inputLock.unlock() }
        do {
            try inputPipe.fileHandleForWriting.write(contentsOf: Data(line.utf8))
            onStderr?("Engine sent \(command)\n")
        } catch {
            onStderr?("Engine write failed: \(error.localizedDescription)\n")
        }
    }

    func shutdown() {
        send("shutdown")
        queue.asyncAfter(deadline: .now() + 30) { [weak self] in
            guard let self, self.process.isRunning else { return }
            self.process.terminate()
        }
    }

    private func startReaders() {
        let output = outputPipe.fileHandleForReading
        outputReadQueue.async { [weak self] in
            Self.read(output, chunkSize: 64 * 1024) { [weak self] data in
                self?.consume(data)
            }
        }

        let errors = errorPipe.fileHandleForReading
        errorReadQueue.async { [weak self] in
            Self.read(errors, chunkSize: 16 * 1024) { [weak self] data in
                if let message = String(data: data, encoding: .utf8) {
                    self?.onStderr?(message)
                }
            }
        }
    }

    private static func read(
        _ handle: FileHandle,
        chunkSize: Int,
        consume: (Data) -> Void
    ) {
        var buffer = [UInt8](repeating: 0, count: chunkSize)
        while true {
            let count = buffer.withUnsafeMutableBytes { bytes in
                Darwin.read(handle.fileDescriptor, bytes.baseAddress, bytes.count)
            }
            if count > 0 {
                consume(Data(buffer.prefix(count)))
            } else if count == 0 {
                return
            } else if errno != EINTR {
                return
            }
        }
    }

    private func consume(_ data: Data) {
        guard !data.isEmpty else { return }
        queue.async { [weak self] in
            guard let self else { return }
            self.outputBuffer.append(data)
            while let newline = self.outputBuffer.firstIndex(of: 0x0A) {
                let line = Data(self.outputBuffer[..<newline])
                self.outputBuffer.removeSubrange(...newline)
                guard !line.isEmpty,
                      let event = try? JSONDecoder().decode(EngineEnvelope.self, from: line) else { continue }
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

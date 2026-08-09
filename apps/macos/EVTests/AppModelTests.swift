import Foundation
import Testing
@testable import EV

private final class FakeEngine: EngineTransport, @unchecked Sendable {
    var onEvent: (@Sendable (EngineEnvelope) -> Void)?
    var onTermination: (@Sendable (Int32) -> Void)?
    var onStderr: (@Sendable (String) -> Void)?
    var isRunning = false
    var commands: [String] = []

    func start() throws { isRunning = true }

    func send(_ command: String, payload: [String: Any]) {
        commands.append(command)
    }

    func shutdown() { isRunning = false }
}

private struct AllowedPermission: MicrophonePermissionProviding {
    var state: MicrophonePermissionState { .authorized }
    func request(_ completion: @escaping @Sendable (Bool) -> Void) { completion(true) }
}

private func envelope(_ type: String, _ payload: String) throws -> EngineEnvelope {
    let json = """
    {"version":1,"request_id":null,"type":"\(type)","timestamp":"2026-01-01T00:00:00Z","payload":\(payload)}
    """
    return try JSONDecoder().decode(EngineEnvelope.self, from: Data(json.utf8))
}

@MainActor
@Test func appModelRetainsFinalAndTracksBackgroundProcessing() throws {
    let engine = FakeEngine()
    let model = AppModel(engine: engine, permissionProvider: AllowedPermission())
    model.handle(try envelope("engine_state", #"{"state":"listening"}"#))
    model.handle(try envelope("capture_started", #"{"sample_rate":16000,"channels":1}"#))
    model.handle(try envelope("speech_started", #"{"segment_id":"seg"}"#))
    model.handle(try envelope("transcript_partial", #"{"segment_id":"seg","text":"你好"}"#))
    model.handle(try envelope("segment_processing", #"{"segment_id":"seg","phase":"finalizing"}"#))

    #expect(model.captureReady)
    #expect(model.partialText == "你好")
    #expect(model.isProcessing)

    model.handle(
        try envelope(
            "segment_committed",
            #"{"id":"seg","started_at":"2026-01-01","duration_ms":1000,"audio_path":"/tmp/seg.wav","transcript_raw":"你好","transcript_final":"你好世界","speaker_label":"unknown","speaker_score":null,"wake_detected":0,"query_candidate":0,"query_text":null}"#
        )
    )

    #expect(model.partialText.isEmpty)
    #expect(model.lastFinalText == "你好世界")
    #expect(!model.isProcessing)
    #expect(model.displayTranscript == "你好世界")
}

@MainActor
@Test func authorizedMicrophoneStartsListening() throws {
    let engine = FakeEngine()
    let model = AppModel(engine: engine, permissionProvider: AllowedPermission())
    model.runtimeReady = true
    model.models = [
        ModelStatus(.object(["key": .string("vad"), "status": .string("ready")])),
        ModelStatus(.object(["key": .string("asr_streaming"), "status": .string("ready")])),
        ModelStatus(.object(["key": .string("asr_final"), "status": .string("ready")])),
        ModelStatus(.object(["key": .string("speaker"), "status": .string("ready")])),
    ].compactMap { $0 }
    model.handle(
        try envelope(
            "device_list",
            #"{"devices":[{"index":1,"name":"Test Mic","is_default":true}]}"#
        )
    )
    model.toggleListening()
    #expect(engine.commands.contains("start_listening"))
    #expect(model.engineState == .loading)

    model.toggleListening()
    #expect(engine.commands.contains("stop_listening"))
    #expect(model.engineState == .stopping)
}

@Test func realEngineClientCompletesHandshake() async throws {
    let support = FileManager.default.temporaryDirectory
        .appendingPathComponent("ev-engine-test-\(UUID().uuidString)", isDirectory: true)
    let client = EngineClient(supportDirectory: support)
    try await confirmation("runtime status received") { confirmed in
        client.onEvent = { event in
            if event.type == "runtime_status" { confirmed() }
        }
        try client.start()
        client.send("get_status")
        try await Task.sleep(for: .seconds(2))
        client.shutdown()
    }
}

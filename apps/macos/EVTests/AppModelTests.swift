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
    model.slotAssignments = [
        SlotAssignment(.object([
            "slot": .string("vad"),
            "model_key": .string("fsmn-vad"),
            "enabled": .bool(true),
            "status": .object(["ready": .bool(true)]),
        ])),
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

@MainActor
@Test func deviceListDoesNotOverrideManualSelectionWhenSystemDefaultDiffers() throws {
    // 闪烁 bug 根因: 之前未监听时若系统默认设备 != 手动选择的, 会被强制覆盖回默认
    // 修复后: 只要手动选择的设备名仍然存在于设备列表, 就保持不动 (不管是否 is_default)
    let engine = FakeEngine()
    let model = AppModel(engine: engine, permissionProvider: AllowedPermission())
    // 第一次: 2 个设备, MacBook 是默认
    model.handle(
        try envelope(
            "device_list",
            #"{"devices":[{"index":0,"name":"MacBook Microphone","is_default":true},{"index":2,"name":"DJI Mic Mini RX","is_default":false}]}"#
        )
    )
    // 验证默认初始是 "使用系统默认" (空串)
    #expect(model.selectedDevice == AppModel.systemDefaultDeviceTag)
    // 用户手动切换到 DJI
    model.selectedDevice = "DJI Mic Mini RX"
    // 触发 Combine sink 同步 -> 会发 set_device
    #expect(engine.commands.filter { $0 == "set_device" }.count == 1)
    // 第二次 device_list (每 5s 刷新一次): 设备列表完全一样, Mac 仍然是默认
    model.handle(
        try envelope(
            "device_list",
            #"{"devices":[{"index":0,"name":"MacBook Microphone","is_default":true},{"index":2,"name":"DJI Mic Mini RX","is_default":false}]}"#
        )
    )
    // 关键断言: 手动选择的 DJI 没有被闪烁覆盖回 MacBook
    #expect(model.selectedDevice == "DJI Mic Mini RX")
    // 再刷一次,仍然保持
    model.handle(
        try envelope(
            "device_list",
            #"{"devices":[{"index":0,"name":"MacBook Microphone","is_default":true},{"index":2,"name":"DJI Mic Mini RX","is_default":false}]}"#
        )
    )
    #expect(model.selectedDevice == "DJI Mic Mini RX")
}

@MainActor
@Test func deviceListFallsBackWhenSelectedDeviceRemoved() throws {
    // 用户选了 DJI, 拔掉 USB -> 下一次 device_list 里没有 DJI -> 自动回退到"使用系统默认"
    let engine = FakeEngine()
    let model = AppModel(engine: engine, permissionProvider: AllowedPermission())
    model.handle(
        try envelope(
            "device_list",
            #"{"devices":[{"index":0,"name":"MacBook Microphone","is_default":true},{"index":2,"name":"DJI Mic Mini RX","is_default":false}]}"#
        )
    )
    model.selectedDevice = "DJI Mic Mini RX"
    #expect(model.selectedDevice == "DJI Mic Mini RX")
    // DJI 被拔出
    model.handle(
        try envelope(
            "device_list",
            #"{"devices":[{"index":0,"name":"MacBook Microphone","is_default":true}]}"#
        )
    )
    #expect(model.selectedDevice == AppModel.systemDefaultDeviceTag)
    // 验证回退不会乱发额外命令 (set_device 已经在手动赋值时发过 1 次; fallback 到 "" 会再触发 sink 发 1 次,共 2 次)
    let setCount = engine.commands.filter { $0 == "set_device" }.count
    #expect(setCount == 2)
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

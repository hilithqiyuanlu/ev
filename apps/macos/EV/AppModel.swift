import AppKit
import Foundation
import ServiceManagement

enum EngineState: String {
    case stopped, loading, listening, speech, stopping, error

    var title: String {
        switch self {
        case .stopped: "停止"
        case .loading: "加载"
        case .listening: "监听"
        case .speech: "正在说话"
        case .stopping: "停止中"
        case .error: "错误"
        }
    }

    var symbol: String {
        switch self {
        case .stopped: "mic.slash"
        case .loading, .stopping: "hourglass"
        case .listening: "waveform"
        case .speech: "waveform.circle.fill"
        case .error: "exclamationmark.triangle"
        }
    }
}

@MainActor
final class AppModel: ObservableObject {
    @Published var engineState: EngineState = .stopped
    @Published var devices: [AudioDevice] = []
    @Published var selectedDevice = ""
    @Published var audioLevel = 0.0
    @Published var partialText = ""
    @Published var segments: [Segment] = []
    @Published var queries: [QueryItem] = []
    @Published var models: [ModelStatus] = []
    @Published var runtimeReady = false
    @Published var runtimeLabel = "检查中"
    @Published var downloadProgress = 0.0
    @Published var downloadLabel = ""
    @Published var enrollmentCompleted = 0
    @Published var enrollmentTotal = 8
    @Published var enrollmentStatus = "尚未开始"
    @Published var errorMessage: String?
    @Published var lastEngineLog = ""
    @Published var speakerFilter = ""
    @Published var queryOnly = false
    @Published var dateFilter = ""
    @Published var userThreshold = UserDefaults.standard.object(forKey: "userThreshold") as? Double ?? 0.72
    @Published var nonUserThreshold = UserDefaults.standard.object(forKey: "nonUserThreshold") as? Double ?? 0.45
    @Published var launchAtLogin = SMAppService.mainApp.status == .enabled

    let audioPlayer = AudioPlayer()
    private let engine = EngineClient()

    var isListening: Bool {
        [.loading, .listening, .speech, .stopping].contains(engineState)
    }

    var allModelsReady: Bool {
        models.count == 4 && models.allSatisfy(\.ready)
    }

    var applicationSupportPath: String {
        let base = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true
        )
        return base?.appendingPathComponent("EV").path ?? ""
    }

    init() {
        engine.onEvent = { [weak self] event in
            Task { @MainActor in self?.handle(event) }
        }
        engine.onTermination = { [weak self] code in
            Task { @MainActor in
                self?.engineState = code == 0 ? .stopped : .error
                if code != 0 { self?.errorMessage = "语音引擎已退出（\(code)）" }
            }
        }
        engine.onStderr = { [weak self] message in
            Task { @MainActor in self?.lastEngineLog = message }
        }
        do {
            try engine.start()
            // 等待 Python engine 完成 stdin/stdout 初始化，再发首批请求。
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
                self?.refreshAll()
            }
        } catch {
            engineState = .error
            errorMessage = error.localizedDescription
        }
    }

    func refreshAll() {
        engine.send("get_status")
        engine.send("list_devices")
        engine.send("verify_models")
        loadHistory()
    }

    func toggleListening() {
        if isListening {
            engine.send("stop_listening")
        } else {
            engine.send(
                "start_listening",
                payload: [
                    "device": selectedDevice,
                    "user_threshold": userThreshold,
                    "non_user_threshold": nonUserThreshold,
                ]
            )
        }
    }

    func loadHistory() {
        engine.send(
            "list_segments",
            payload: [
                "limit": 200,
                "speaker_label": speakerFilter,
                "query_only": queryOnly,
                "date": dateFilter,
            ]
        )
    }

    func submitQuery(_ text: String) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        engine.send("submit_manual_query", payload: ["text": text])
    }

    func deleteSegment(_ id: String) {
        engine.send("delete_segment", payload: ["segment_id": id])
    }

    func deleteAllSegments() {
        engine.send("delete_all_segments")
    }

    func deleteQuery(_ id: String) {
        engine.send("delete_query", payload: ["query_id": id])
    }

    func deleteAllQueries() {
        engine.send("delete_all_queries")
    }

    func downloadModels() {
        downloadProgress = 0
        engine.send("download_models")
    }

    func verifyModels() {
        engine.send("verify_models")
    }

    func cancelDownload() {
        engine.send("cancel_download")
    }

    func beginEnrollment() {
        enrollmentCompleted = 0
        enrollmentTotal = 8
        engine.send(
            "begin_enrollment",
            payload: ["segments": enrollmentTotal, "device": selectedDevice]
        )
    }

    func captureEnrollmentSample() {
        engine.send("capture_enrollment_sample")
    }

    func cancelEnrollment() {
        engine.send("cancel_enrollment")
    }

    func saveThresholds() {
        UserDefaults.standard.set(userThreshold, forKey: "userThreshold")
        UserDefaults.standard.set(nonUserThreshold, forKey: "nonUserThreshold")
    }

    func setLaunchAtLogin(_ enabled: Bool) {
        do {
            if enabled { try SMAppService.mainApp.register() }
            else { try SMAppService.mainApp.unregister() }
            launchAtLogin = enabled
        } catch {
            launchAtLogin = false
            errorMessage = "无法修改开机启动：\(error.localizedDescription)"
        }
    }

    func openApplicationSupport() {
        NSWorkspace.shared.open(URL(fileURLWithPath: applicationSupportPath, isDirectory: true))
    }

    func openLogs() {
        let path = URL(fileURLWithPath: applicationSupportPath).appendingPathComponent("logs")
        NSWorkspace.shared.open(path)
    }

    func openInFinder(_ path: String) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }

    func shutdown() {
        engine.shutdown()
    }

    private func handle(_ event: EngineEnvelope) {
        switch event.type {
        case "engine_state":
            engineState = EngineState(rawValue: event.payload["state"]?.string ?? "stopped") ?? .error
        case "device_list":
            devices = event.payload["devices"]?.array?.compactMap(AudioDevice.init) ?? []
            if selectedDevice.isEmpty { selectedDevice = devices.first(where: \.isDefault)?.name ?? devices.first?.name ?? "" }
        case "audio_level":
            audioLevel = min(max((event.payload["rms"]?.double ?? 0) * 10, 0), 1)
        case "transcript_partial":
            partialText = event.payload["text"]?.string ?? ""
        case "segment_committed":
            if let segment = Segment(event.payload) {
                segments.removeAll { $0.id == segment.id }
                segments.insert(segment, at: 0)
                partialText = ""
            }
        case "segment_list":
            segments = event.payload["segments"]?.array?.compactMap { $0.object.flatMap(Segment.init) } ?? []
            queries = event.payload["queries"]?.array?.compactMap(QueryItem.init) ?? []
        case "segment_deleted", "segments_deleted", "query_deleted", "queries_deleted":
            loadHistory()
        case "query_candidate":
            loadHistory()
        case "model_status":
            handleModelStatus(event.payload)
        case "runtime_status":
            runtimeReady = event.payload["ready"]?.bool ?? false
            runtimeLabel = runtimeReady ? "FunASR 与 PyTorch 已安装" : "缺少 FunASR 或 PyTorch（开发版需在 .venv 安装）"
        case "download_progress":
            let downloaded = event.payload["total_downloaded"]?.double ?? 0
            let total = event.payload["total_size"]?.double ?? 1
            downloadProgress = total > 0 ? downloaded / total : 0
            downloadLabel = event.payload["key"]?.string ?? ""
        case "enrollment_progress":
            enrollmentCompleted = Int(event.payload["completed"]?.double ?? 0)
            enrollmentTotal = max(Int(event.payload["total"]?.double ?? 8), enrollmentCompleted)
            enrollmentStatus = enrollmentTitle(event.payload["status"]?.string ?? "")
        case "error":
            errorMessage = event.payload["message"]?.string ?? "未知错误"
        default:
            break
        }
    }

    private func handleModelStatus(_ payload: [String: JSONValue]) {
        if let values = payload["models"]?.array {
            models = values.compactMap(ModelStatus.init)
            return
        }
        if let status = ModelStatus(.object(payload)) {
            models.removeAll { $0.key == status.key }
            models.append(status)
            models.sort { $0.key < $1.key }
        }
        if payload["status"]?.string == "complete" {
            downloadProgress = 1
            engine.send("verify_models")
        }
    }

    private func enrollmentTitle(_ status: String) -> String {
        switch status {
        case "ready": "准备录入"
        case "recording": "正在录音 4 秒"
        case "sample_complete": "本段完成"
        case "complete": "声纹录入完成"
        case "cancelled": "已取消"
        default: status
        }
    }
}

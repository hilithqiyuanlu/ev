import AppKit
import AVFoundation
import Combine
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

enum MicrophonePermissionState: String {
    case notDetermined, authorized, denied, restricted

    var title: String {
        switch self {
        case .notDetermined: "尚未请求"
        case .authorized: "已允许"
        case .denied: "已拒绝"
        case .restricted: "受系统限制"
        }
    }
}

protocol MicrophonePermissionProviding {
    var state: MicrophonePermissionState { get }
    func request(_ completion: @escaping @Sendable (Bool) -> Void)
}

struct SystemMicrophonePermission: MicrophonePermissionProviding {
    var state: MicrophonePermissionState {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: .authorized
        case .denied: .denied
        case .restricted: .restricted
        case .notDetermined: .notDetermined
        @unknown default: .restricted
        }
    }

    func request(_ completion: @escaping @Sendable (Bool) -> Void) {
        AVCaptureDevice.requestAccess(for: .audio, completionHandler: completion)
    }
}

@MainActor
final class AppModel: ObservableObject {
    @Published var engineState: EngineState = .stopped
    @Published var devices: [AudioDevice] = []
    @Published var selectedDevice = ""
    @Published var audioLevel = 0.0
    @Published var partialText = ""
    @Published var lastFinalText = ""
    @Published var captureReady = false
    @Published var processingSegmentIDs: Set<String> = []
    @Published var segments: [Segment] = []
    @Published var queries: [QueryItem] = []
    @Published var historyItems: [HistoryItem] = []
    @Published var models: [ModelStatus] = []
    @Published var runtimeReady = false
    @Published var runtimeLabel = "检查中"
    @Published var downloadProgress = 0.0
    @Published var downloadLabel = ""
    @Published var voiceProfile = VoiceProfileState.empty
    @Published var voiceSamples: [VoiceSample] = []
    @Published var manualEnrollStatus = "idle" // idle, recording, processing, done, failed
    @Published var manualEnrollError: String?
    @Published var manualEnrollDuration: Double = 3.0
    @Published var errorMessage: String?
    @Published var lastEngineLog = ""
    @Published var speakerFilter = ""
    @Published var queryOnly = false
    @Published var dateFilter = ""
    @Published var speakerThreshold: Double
    @Published var autoLearnEnabled = UserDefaults.standard.object(forKey: "autoLearnEnabled") as? Bool ?? true
    @Published var launchAtLogin = SMAppService.mainApp.status == .enabled
    @Published var ctrlTToggleListening = UserDefaults.standard.object(forKey: "ctrlTToggleListening") as? Bool ?? true
    @Published var microphonePermission: MicrophonePermissionState
    @Published var hasCompletedOnboarding: Bool
    @Published var isVerifyingModels = false
    @Published var isLoadingHistory = false
    @Published var showVerificationDone = false

    let audioPlayer = AudioPlayer()
    private let engine: EngineTransport
    private let permissionProvider: MicrophonePermissionProviding
    private var didInitialRefresh = false
    private var deviceRefreshTimer: Timer?
    private var cancellables = Set<AnyCancellable>()

    var onboardingChecks: (models: Bool, permission: Bool) {
        (
            allModelsReady,
            microphonePermission == .authorized
        )
    }

    var isOnboardingComplete: Bool {
        let checks = onboardingChecks
        return checks.models && checks.permission
    }

    var isListening: Bool {
        [.loading, .listening, .speech, .stopping].contains(engineState)
    }

    var allModelsReady: Bool {
        models.count == 4 && models.allSatisfy(\.ready)
    }

    var isProcessing: Bool { !processingSegmentIDs.isEmpty }

    var canStartListening: Bool {
        allModelsReady && runtimeReady && !devices.isEmpty &&
            microphonePermission != .denied && microphonePermission != .restricted
    }

    var activityTitle: String {
        if engineState == .error { return "语音引擎出错" }
        if engineState == .loading { return "正在加载语音模型" }
        if engineState == .stopping { return "正在停止监听" }
        if engineState == .speech { return "检测到语音" }
        if isProcessing { return "正在生成转写" }
        if engineState == .listening { return captureReady ? "正在监听" : "正在连接麦克风" }
        return "尚未监听"
    }

    var activityDetail: String {
        if engineState == .speech { return "继续说话，停顿后会自动生成终稿" }
        if isProcessing { return "正在完成终稿、声纹判断和保存" }
        if engineState == .listening { return "所有检测到的人声都会保存在本机" }
        if engineState == .loading { return "首次加载可能需要几秒" }
        return "选择麦克风后开始监听"
    }

    var activitySymbol: String {
        if isProcessing && engineState != .speech { return "text.bubble.fill" }
        return engineState.symbol
    }

    var displayTranscript: String {
        if !partialText.isEmpty { return partialText }
        if !lastFinalText.isEmpty { return lastFinalText }
        return "等待语音"
    }

    var applicationSupportPath: String {
        let base = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true
        )
        return base?.appendingPathComponent("EV").path ?? ""
    }

    init(
        engine: EngineTransport = EngineClient(),
        permissionProvider: MicrophonePermissionProviding = SystemMicrophonePermission()
    ) {
        self.engine = engine
        self.permissionProvider = permissionProvider
        self.microphonePermission = permissionProvider.state
        self.hasCompletedOnboarding = UserDefaults.standard.bool(forKey: "hasCompletedOnboarding")
        // Migrate threshold: prefer new key, fallback to old userThreshold key, default 0.50
        let defaults = UserDefaults.standard
        if let saved = defaults.object(forKey: "speakerThreshold") as? Double {
            self.speakerThreshold = saved
        } else if let oldUser = defaults.object(forKey: "userThreshold") as? Double {
            self.speakerThreshold = oldUser
        } else {
            self.speakerThreshold = 0.50
        }
        engine.onEvent = { [weak self] event in
            Task { @MainActor in self?.handle(event) }
        }
        engine.onTermination = { [weak self] code in
            Task { @MainActor in
                if let self {
                    self.deviceRefreshTimer?.invalidate()
                    self.deviceRefreshTimer = nil
                    self.lastEngineLog = String(
                        (self.lastEngineLog + "Engine exited with code \(code)\n").suffix(8_192)
                    )
                }
                self?.engineState = code == 0 ? .stopped : .error
                if code != 0 { self?.errorMessage = "语音引擎已退出（\(code)）" }
            }
        }
        engine.onStderr = { [weak self] message in
            Task { @MainActor in
                guard let self else { return }
                self.lastEngineLog = String((self.lastEngineLog + message).suffix(8_192))
            }
        }
        do {
            try engine.start()
            refreshAll()
            if ctrlTToggleListening { registerGlobalHotkey() }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.refreshAll()
            }
            // 每2秒刷新一次设备列表，检测设备连接/断开
            deviceRefreshTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
                Task { @MainActor in self?.refreshDevices() }
            }
            // Listen for device changes: when user selects different device, notify engine immediately
            $selectedDevice
                .dropFirst()  // Skip initial value
                .removeDuplicates()
                .sink { [weak self] newDevice in
                    guard let self, !newDevice.isEmpty else { return }
                    // Send set_device command - engine will restart listening if active
                    self.engine.send("set_device", payload: ["device": newDevice])
                }
                .store(in: &cancellables)
        } catch {
            engineState = .error
            errorMessage = error.localizedDescription
        }
    }

    func refreshDevices() {
        engine.send("list_devices")
    }

    func refreshAll() {
        engine.send("get_status")
        engine.send("list_devices")
        engine.send("verify_models")
        loadHistory()
        loadVoiceSamples()
    }

    func toggleListening() {
        if isListening {
            engineState = .stopping
            engine.send("stop_listening")
        } else {
            requestPermissionAndStart()
        }
    }

    private func requestPermissionAndStart() {
        microphonePermission = permissionProvider.state
        switch microphonePermission {
        case .authorized:
            startListening()
        case .notDetermined:
            permissionProvider.request { [weak self] granted in
                Task { @MainActor in
                    self?.microphonePermission = granted ? .authorized : .denied
                    if granted { self?.startListening() }
                    else { self?.errorMessage = "未获得麦克风权限，请在系统设置中允许 EV 使用麦克风。" }
                }
            }
        case .denied, .restricted:
            errorMessage = "麦克风权限不可用，请在系统设置中检查隐私与安全性。"
        }
    }

    private func startListening() {
        guard canStartListening else {
            errorMessage = devices.isEmpty ? "未发现输入设备。" : "模型或 Python 运行时尚未就绪。"
            return
        }
        captureReady = false
        partialText = ""
        engineState = .loading
        engine.send(
            "start_listening",
            payload: [
                "device": selectedDevice,
                "threshold": speakerThreshold,
                "auto_learn": autoLearnEnabled,
            ]
        )
    }

    func loadHistory() {
        isLoadingHistory = true
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
        isVerifyingModels = true
        showVerificationDone = false
        engine.send("verify_models")
    }

    func cancelDownload() {
        engine.send("cancel_download")
    }

    func loadVoiceSamples() {
        engine.send("list_voice_samples", payload: ["limit": 50])
    }

    func deleteVoiceSample(_ id: String) {
        engine.send("delete_voice_sample", payload: ["sample_id": id])
    }

    func promoteVoiceSample(_ id: String) {
        engine.send("promote_voice_sample", payload: ["sample_id": id])
    }

    func resetVoiceProfile() {
        engine.send("reset_voice_profile")
    }

    func setAutoLearn(_ enabled: Bool) {
        autoLearnEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "autoLearnEnabled")
        engine.send("set_voice_learning", payload: ["enabled": enabled])
    }

    func captureManualSample() {
        guard manualEnrollStatus != "recording" && manualEnrollStatus != "processing" else { return }
        manualEnrollStatus = "recording"
        manualEnrollError = nil
        engine.send("capture_manual_sample", payload: ["duration_sec": manualEnrollDuration])
    }

    func saveThresholds() {
        UserDefaults.standard.set(speakerThreshold, forKey: "speakerThreshold")
        // Send to engine immediately - takes effect without restarting listening
        engine.send(
            "set_thresholds",
            payload: [
                "threshold": speakerThreshold,
            ]
        )
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

    func setCtrlTToggleListening(_ enabled: Bool) {
        ctrlTToggleListening = enabled
        UserDefaults.standard.set(enabled, forKey: "ctrlTToggleListening")
        if enabled { registerGlobalHotkey() } else { unregisterGlobalHotkey() }
    }

    private var localMonitor: Any?
    private var globalMonitor: Any?

    private func registerGlobalHotkey() {
        unregisterGlobalHotkey()
        localMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            self?.handleHotkey(event)
            return event
        }
        globalMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            self?.handleHotkey(event)
        }
    }

    private func unregisterGlobalHotkey() {
        if let localMonitor { NSEvent.removeMonitor(localMonitor); self.localMonitor = nil }
        if let globalMonitor { NSEvent.removeMonitor(globalMonitor); self.globalMonitor = nil }
    }

    private func handleHotkey(_ event: NSEvent) {
        guard ctrlTToggleListening else { return }
        let modifiers = event.modifierFlags.intersection([.control, .command, .option, .shift])
        if modifiers == .control && event.keyCode == 17 {
            toggleListening()
        }
    }

    func openApplicationSupport() {
        NSWorkspace.shared.open(URL(fileURLWithPath: applicationSupportPath, isDirectory: true))
    }

    func openLogs() {
        let path = URL(fileURLWithPath: applicationSupportPath).appendingPathComponent("logs")
        NSWorkspace.shared.open(path)
    }

    func openMicrophoneSettings() {
        guard let url = URL(
            string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
        ) else { return }
        NSWorkspace.shared.open(url)
    }

    func openInFinder(_ path: String) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }

    func shutdown() {
        engine.shutdown()
    }

    func quitApplication() {
        shutdown()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            NSApp.terminate(nil)
        }
    }

    func handle(_ event: EngineEnvelope) {
        switch event.type {
        case "engine_state":
            engineState = EngineState(rawValue: event.payload["state"]?.string ?? "stopped") ?? .error
            if engineState == .stopped || engineState == .error { captureReady = false }
            if event.requestID == nil && !didInitialRefresh {
                didInitialRefresh = true
                refreshAll()
            }
        case "device_list":
            devices = event.payload["devices"]?.array?.compactMap(AudioDevice.init) ?? []
            let deviceNames = Set(devices.map(\.name))
            let currentDeviceExists = !selectedDevice.isEmpty && deviceNames.contains(selectedDevice)
            if !currentDeviceExists {
                // 当前选中设备不在列表中（断开连接）或未选择，自动选择默认设备
                selectedDevice = devices.first(where: \.isDefault)?.name ?? devices.first?.name ?? ""
            } else if !isListening, let defaultDevice = devices.first(where: \.isDefault), defaultDevice.name != selectedDevice {
                // 未监听时，如果系统默认设备变了，自动跟随系统默认
                selectedDevice = defaultDevice.name
            }
            completeOnboardingIfNeeded()
        case "audio_level":
            audioLevel = min(max((event.payload["rms"]?.double ?? 0) * 10, 0), 1)
        case "capture_started":
            captureReady = true
        case "speech_started":
            partialText = ""
        case "transcript_partial":
            partialText = event.payload["text"]?.string ?? ""
        case "segment_processing":
            if let id = event.payload["segment_id"]?.string { processingSegmentIDs.insert(id) }
        case "segment_committed":
            if let segment = Segment(event.payload) {
                segments.removeAll { $0.id == segment.id }
                segments.insert(segment, at: 0)
                processingSegmentIDs.remove(segment.id)
                lastFinalText = segment.transcript
                partialText = ""
                rebuildHistoryItems()
            }
        case "segment_failed":
            if let id = event.payload["segment_id"]?.string { processingSegmentIDs.remove(id) }
            errorMessage = event.payload["message"]?.string ?? "语音段处理失败"
        case "segment_list":
            segments = event.payload["segments"]?.array?.compactMap { $0.object.flatMap(Segment.init) } ?? []
            queries = event.payload["queries"]?.array?.compactMap(QueryItem.init) ?? []
            rebuildHistoryItems()
            isLoadingHistory = false
        case "segment_deleted", "segments_deleted", "query_deleted", "queries_deleted":
            loadHistory()
            // Deleting segments may cascade-delete auto voice samples - refresh
            loadVoiceSamples()
        case "query_candidate":
            loadHistory()
        case "model_status":
            handleModelStatus(event.payload)
            if event.payload["models"] != nil {
                isVerifyingModels = false
                showVerificationDone = true
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: 2_000_000_000)
                    self.showVerificationDone = false
                }
                completeOnboardingIfNeeded()
            }
        case "runtime_status":
            runtimeReady = event.payload["ready"]?.bool ?? false
            runtimeLabel = runtimeReady ? "FunASR 与 PyTorch 已安装" : "缺少 FunASR 或 PyTorch（开发版需在 .venv 安装）"
        case "download_progress":
            let downloaded = event.payload["total_downloaded"]?.double ?? 0
            let total = event.payload["total_size"]?.double ?? 1
            downloadProgress = total > 0 ? downloaded / total : 0
            downloadLabel = event.payload["key"]?.string ?? ""
        case "profile_status":
            voiceProfile = VoiceProfileState(.object(event.payload))
            if let autoLearn = event.payload["auto_learn"]?.bool {
                autoLearnEnabled = autoLearn
            } else if let autoLearn = event.payload["auto_learn"]?.double {
                autoLearnEnabled = autoLearn != 0
            }
            loadVoiceSamples()
        case "voice_samples":
            voiceSamples = event.payload["samples"]?.array?.compactMap(VoiceSample.init) ?? []
        case "voice_sample_added", "voice_sample_promoted", "voice_sample_deleted", "voice_profile_reset":
            loadVoiceSamples()
            // Also refresh profile status after sample changes
            engine.send("get_status")
        case "manual_sample_status":
            let status = event.payload["status"]?.string ?? "failed"
            manualEnrollStatus = status
            manualEnrollError = event.payload["error"]?.string
            if status == "done" || status == "failed" {
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
                    if self?.manualEnrollStatus == status {
                        self?.manualEnrollStatus = "idle"
                        self?.manualEnrollError = nil
                    }
                }
            }
        case "error":
            if engineState == .loading || engineState == .stopping {
                engineState = .error
            }
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

    func completeOnboardingIfNeeded() {
        guard !hasCompletedOnboarding, isOnboardingComplete else { return }
        hasCompletedOnboarding = true
        UserDefaults.standard.set(true, forKey: "hasCompletedOnboarding")
    }

    func resetOnboarding() {
        hasCompletedOnboarding = false
        UserDefaults.standard.set(false, forKey: "hasCompletedOnboarding")
    }

    private func rebuildHistoryItems() {
        let segmentIDs = Set(segments.map { $0.id })

        let filteredSegments = segments.filter { segment in
            if !speakerFilter.isEmpty {
                guard segment.speakerLabel == speakerFilter else { return false }
            }
            if !dateFilter.isEmpty {
                guard segment.startedAt.hasPrefix(dateFilter) else { return false }
            }
            if queryOnly {
                guard segment.queryCandidate else { return false }
            }
            return true
        }

        let filteredQueries = queries.filter { query in
            if query.source == "voice" && (query.segmentId.map { segmentIDs.contains($0) } ?? false) {
                return false
            }
            if !dateFilter.isEmpty {
                guard query.createdAt.hasPrefix(dateFilter) else { return false }
            }
            return true
        }

        let segmentItems = filteredSegments.map { HistoryItem.segment($0) }
        let queryItems = filteredQueries.map { HistoryItem.query($0) }
        historyItems = (segmentItems + queryItems).sorted { $0.sortDate > $1.sortDate }
    }
}

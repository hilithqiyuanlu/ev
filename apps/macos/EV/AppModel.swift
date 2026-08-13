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
    /// 空字符串表示「使用系统默认输入设备」，和后端 set_device 的约定保持一致
    static let systemDefaultDeviceTag = ""
    private static let selectedDeviceDefaultsKey = "selectedDevice"
    private static let selectedDeviceLabelDefaultsKey = "selectedDeviceLabel"

    @Published var engineState: EngineState = .stopped
    @Published var devices: [AudioDevice] = []
    /// 选中的设备名；空串 = 使用系统默认
    @Published var selectedDevice = AppModel.systemDefaultDeviceTag
    @Published var audioLevel = 0.0
    @Published var rawRmsDb: Double = -100.0   // 原始(预处理前) RMS dBFS, 调试远场用
    @Published var agcGain: Double = 1.0       // 当前 AGC 增益倍数

    /// 设备选择 Menu 显示用: 所有真实设备，系统默认设备标注「（默认）」
    var devicePickerItems: [(tag: String, label: String, isDefault: Bool)] {
        typealias Item = (tag: String, label: String, isDefault: Bool)
        return devices.map { dev in
            (
                tag: dev.name,
                label: dev.isDefault ? "\(dev.name)（默认）" : dev.name,
                isDefault: dev.isDefault
            )
        }
    }

    /// 当前选中设备的折叠显示名；未选中（空串 = 系统默认）时显示默认设备名并标记「默认」
    var selectedDeviceLabel: String {
        if selectedDevice.isEmpty {
            if let def = devices.first(where: \.isDefault) {
                return "\(def.name)（默认）"
            }
            return devices.first?.name ?? "无可用设备"
        }
        if let dev = devices.first(where: { $0.name == selectedDevice }) {
            return dev.isDefault ? "\(dev.name)（默认）" : dev.name
        }
        return selectedDevice
    }

    @Published var partialText = ""
    @Published var lastFinalText = ""
    @Published var captureReady = false
    @Published var envCategory: String = ""
    @Published var envCategoryLabel: String = ""
    @Published var envConfidence: Double = 0
    @Published var envActive: Bool = false
    @Published var envLastUpdate: Date? = nil
    @Published var processingSegmentIDs: Set<String> = []
    @Published var segments: [Segment] = []
    @Published var queries: [QueryItem] = []
    @Published var historyItems: [HistoryItem] = []
    @Published var availableModels: [AvailableModel] = []
    @Published var installedModels: [InstalledModel] = []
    @Published var slotAssignments: [SlotAssignment] = []
    @Published var runtimeReady = false
    @Published var runtimeLabel = "检查中"
    @Published var downloadProgress = 0.0
    @Published var downloadLabel = ""
    @Published var downloadingKey: String?
    @Published var downloadProgressByKey: [String: Double] = [:]
    @Published var downloadStageByKey: [String: String] = [:]
    @Published var downloadMessageByKey: [String: String] = [:]
    @Published var downloadSizeByKey: [String: Int64] = [:]
    @Published var downloadSpeedByKey: [String: Double] = [:]
    @Published var downloadIndeterminateByKey: [String: Bool] = [:]
    @Published var downloadSourceByKey: [String: String] = [:]
    @Published var voiceProfile = VoiceProfileState.empty
    @Published var voiceSamples: [VoiceSample] = []
    @Published var pendingSamples: [VoiceSample] = []
    @Published var lexiconWords: [LexiconItem] = []
    @Published var manualEnrollStatus = "idle" // idle, recording, processing, done, failed
    @Published var manualEnrollError: String?
    @Published var manualEnrollDuration: Double = 3.0
    @Published var enrollStatus = "idle" // idle, recording, processing, done, failed
    @Published var enrollLevel: Double = 0.0
    @Published var errorMessage: String?
    @Published var lastEngineLog = ""
    @Published var speakerFilter = ""
    @Published var queryOnly = false
    @Published var dateFilter = ""
    @Published var speakerThreshold: Double
    @Published var autoLearnEnabled = true
    @Published var isLearningSamples = false
    @Published var launchAtLogin = SMAppService.mainApp.status == .enabled
    @Published var ctrlTToggleListening = UserDefaults.standard.object(forKey: "ctrlTToggleListening") as? Bool ?? true
    @Published var microphonePermission: MicrophonePermissionState
    @Published var isVerifyingModels = false
    @Published var isLoadingHistory = false
    @Published var showVerificationDone = false
    @Published var learnedWordsToast: String?
    @Published var showLearnedWordsToast = false

    let audioPlayer = AudioPlayer()
    private let engine: EngineTransport
    private let permissionProvider: MicrophonePermissionProviding
    private var didInitialRefresh = false
    private var receivedInitialDeviceList = false
    private var deviceRefreshTimer: Timer?
    private var cancellables = Set<AnyCancellable>()

    var isListening: Bool {
        [.loading, .listening, .speech, .stopping].contains(engineState)
    }

    var allModelsReady: Bool {
        let activeAssignments = slotAssignments.filter(\.enabled)
        guard !activeAssignments.isEmpty else { return false }
        return activeAssignments.allSatisfy(\.status.ready)
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
        let defaults = UserDefaults.standard
        // Threshold
        if let saved = defaults.object(forKey: "speakerThreshold") as? Double {
            self.speakerThreshold = saved
        } else if let oldUser = defaults.object(forKey: "userThreshold") as? Double {
            self.speakerThreshold = oldUser
        } else {
            self.speakerThreshold = 0.40
        }
        // 选中设备: 从 UserDefaults 恢复, 默认是空串 = 系统默认
        // 等第一次 device_list 到达后再校验: 如果保存的设备名不在当前列表里 → 回退到 系统默认
        let persistedDevice = defaults.string(forKey: Self.selectedDeviceDefaultsKey) ?? ""
        self.selectedDevice = persistedDevice
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
            // 每 5 秒刷新一次设备列表（2s 太急，且会与用户手动选择竞争，改为 5s）
            deviceRefreshTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
                Task { @MainActor in self?.refreshDevices() }
            }
            // 用户改变选择: 立即持久化 + 通知后端（set_device 监听中会返回错误，由 error handler 提示）
            $selectedDevice
                .dropFirst()
                .removeDuplicates()
                .sink { [weak self] newDevice in
                    guard let self else { return }
                    // 持久化
                    UserDefaults.standard.set(newDevice, forKey: Self.selectedDeviceDefaultsKey)
                    // 通知后端: newDevice 可能是空串（系统默认），后端会归一化成 None
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
        engine.send("list_available_models")
        engine.send("list_installed_models")
        loadHistory()
        loadVoiceSamples()
        loadLexicon()
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

    func correctSegment(_ segmentId: String, correctedText: String) {
        engine.send("correct_segment", payload: ["segment_id": segmentId, "corrected_text": correctedText])
    }

    func deleteAllSegments() {
        engine.send("delete_all_segments")
    }

    func deleteQualityRejectedSegments() {
        engine.send("delete_quality_rejected_segments")
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

    func refreshModelCatalog() {
        engine.send("list_available_models")
        engine.send("list_installed_models")
    }

    func installModel(key: String) {
        downloadingKey = key
        downloadProgressByKey[key] = 0
        downloadStageByKey[key] = "listing"
        downloadMessageByKey[key] = "正在连接..."
        downloadSizeByKey[key] = 0
        downloadSpeedByKey[key] = 0
        downloadIndeterminateByKey[key] = true
        downloadSourceByKey[key] = ""
        engine.send("install_model", payload: ["model_key": key])
    }

    func uninstallModel(key: String) {
        engine.send("uninstall_model", payload: ["model_key": key])
    }

    func setActiveModel(slot: String, modelKey: String?) {
        engine.send("set_active_model", payload: [
            "slot": slot,
            "model_key": modelKey ?? ""
        ])
    }

    func reloadRegistry() {
        engine.send("reload_registry")
    }

    func goToModels() {
        NotificationCenter.default.post(name: NSNotification.Name("goModels"), object: nil)
    }

    func loadVoiceSamples() {
        engine.send("list_voice_samples", payload: ["limit": 50])
    }

    func loadLexicon() {
        engine.send("list_lexicon")
    }

    func addLexiconWord(_ word: String) {
        let trimmed = word.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        engine.send("add_lexicon_word", payload: ["word": trimmed])
    }

    func updateLexiconWord(_ id: String, word: String? = nil, weight: Double? = nil, promoteToManual: Bool = false) {
        var payload: [String: Any] = ["id": id]
        if let word { payload["word"] = word }
        if let weight { payload["weight"] = weight }
        if promoteToManual { payload["promote_to_manual"] = true }
        engine.send("update_lexicon_word", payload: payload)
    }

    func deleteLexiconWord(_ id: String) {
        engine.send("delete_lexicon_word", payload: ["id": id])
    }

    func clearAutoLexicon() {
        engine.send("clear_auto_lexicon")
    }

    func learnCorrections() {
        engine.send("learn_corrections")
    }

    func deleteVoiceSample(_ id: String) {
        engine.send("delete_voice_sample", payload: ["sample_id": id])
    }

    func promoteVoiceSample(_ id: String) {
        engine.send("promote_voice_sample", payload: ["sample_id": id])
    }

    func loadPendingVoiceSamples() {
        engine.send("list_pending_voice_samples")
    }

    func confirmPendingVoiceSample(_ id: String) {
        engine.send("confirm_voice_sample", payload: ["sample_id": id])
    }

    func rejectPendingVoiceSample(_ id: String) {
        engine.send("reject_voice_sample", payload: ["sample_id": id])
    }

    func resetVoiceProfile() {
        engine.send("reset_voice_profile")
    }

    func learnVoiceSamples() {
        guard !isLearningSamples else { return }
        isLearningSamples = true
        engine.send("learn_voice_samples")
    }

    /// 手动引导需要录制的样本数量（与后端 voice_learning.onboarding_target 一致）
    var onboardingTarget: Int { 5 }

    var onboardingCount: Int { min(voiceProfile.coreCount, onboardingTarget) }

    var onboardingProgress: Double {
        guard onboardingTarget > 0 else { return 0 }
        return min(Double(onboardingCount) / Double(onboardingTarget), 1.0)
    }

    var needsVoiceOnboarding: Bool { voiceProfile.coreCount < onboardingTarget }

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

    var isEnrolling: Bool {
        enrollStatus == "recording" || enrollStatus == "processing"
    }

    func startVoiceEnrollment() {
        guard !isEnrolling else { return }
        enrollStatus = "recording"
        enrollLevel = 0.0
        engine.send("start_voice_enrollment")
    }

    func stopVoiceEnrollment() {
        guard enrollStatus == "recording" else { return }
        engine.send("stop_voice_enrollment")
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
            if engineState == .stopped || engineState == .error {
                captureReady = false
                audioLevel = 0
                rawRmsDb = -100
                agcGain = 1.0
            }
            if event.requestID == nil && !didInitialRefresh {
                didInitialRefresh = true
                refreshAll()
            }
        case "device_list":
            let newDevices = event.payload["devices"]?.array?.compactMap(AudioDevice.init) ?? []
            let previousNames = Set(devices.map(\.name))
            devices = newDevices
            let deviceNames = Set(newDevices.map(\.name))

            if receivedInitialDeviceList {
                // 新接入且成为系统默认 → 自动切到系统默认（跟随新默认设备）
                if let addedDefault = newDevices.first(where: { !previousNames.contains($0.name) && $0.isDefault }) {
                    selectedDevice = Self.systemDefaultDeviceTag
                    lastEngineLog = String(
                        (lastEngineLog + "已接入新默认输入设备「\(addedDefault.name)」，已切换为使用系统默认\n")
                            .suffix(8_192)
                    )
                }
                // 用户选中的设备被拔出 → 回退到系统默认
                else if selectedDevice != Self.systemDefaultDeviceTag && !deviceNames.contains(selectedDevice) {
                    let removed = selectedDevice
                    selectedDevice = Self.systemDefaultDeviceTag
                    lastEngineLog = String(
                        (lastEngineLog + "输入设备「\(removed)」已断开，已自动切换为使用系统默认\n")
                            .suffix(8_192)
                    )
                }
                // 其余情况 —— 不再强制覆盖用户手动选择（避免闪烁）
            } else {
                // 首次列表到达：校验 UserDefaults 恢复的设备是否仍存在
                receivedInitialDeviceList = true
                if selectedDevice != Self.systemDefaultDeviceTag && !deviceNames.contains(selectedDevice) {
                    let persisted = selectedDevice
                    selectedDevice = Self.systemDefaultDeviceTag
                    lastEngineLog = String(
                        (lastEngineLog + "保存的输入设备「\(persisted)」不在当前列表中，已回退为使用系统默认\n")
                            .suffix(8_192)
                    )
                }
            }
        case "audio_level":
            audioLevel = min(max((event.payload["rms"]?.double ?? 0) * 10, 0), 1)
            if let rawRms = event.payload["raw_rms"]?.double, rawRms > 0 {
                rawRmsDb = 20.0 * log10(max(rawRms, 1e-7))
            } else {
                rawRmsDb = -100
            }
            agcGain = event.payload["gain"]?.double ?? 1.0
        case "capture_started":
            captureReady = true
            envCategory = ""
            envCategoryLabel = ""
            envActive = false
        case "environment_event":
            let cat = event.payload["category"]?.string ?? ""
            envCategory = cat
            envCategoryLabel = envDisplayName(cat)
            envConfidence = event.payload["confidence"]?.double ?? 0
            envActive = envConfidence > 0.3
            envLastUpdate = Date()
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
        case "segment_discarded":
            if let id = event.payload["segment_id"]?.string { processingSegmentIDs.remove(id) }
        case "segment_list":
            segments = event.payload["segments"]?.array?.compactMap { $0.object.flatMap(Segment.init) } ?? []
            queries = event.payload["queries"]?.array?.compactMap(QueryItem.init) ?? []
            rebuildHistoryItems()
            isLoadingHistory = false
        case "segment_deleted", "segments_deleted", "query_deleted", "queries_deleted":
            loadHistory()
            // Segment deletion preserves samples but clears their segment link.
            loadVoiceSamples()
        case "segment_corrected":
            if let changed = event.payload["changed"]?.bool, changed,
               let segmentId = event.payload["segment_id"]?.string {
                // Optimistic update: mark as corrected locally
                segments = segments.map { seg in
                    if seg.id == segmentId {
                        // Re-fetch to get updated transcript
                        return seg  // will be replaced by loadHistory
                    }
                    return seg
                }
            }
            loadHistory()
        case "query_candidate":
            loadHistory()
        case "model_status":
            if event.payload["models"] != nil {
                isVerifyingModels = false
                showVerificationDone = true
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: 2_000_000_000)
                    self.showVerificationDone = false
                }
            }
            if event.payload["status"]?.string == "complete" {
                downloadProgress = 1
                engine.send("verify_models")
            }
        case "available_models":
            availableModels = event.payload["models"]?.array?.compactMap(AvailableModel.init) ?? []
        case "installed_models":
            installedModels = event.payload["installed"]?.array?.compactMap(InstalledModel.init) ?? []
            slotAssignments = event.payload["slots"]?.array?.compactMap(SlotAssignment.init) ?? []
            if event.payload["all_ready"]?.bool != nil {
                isVerifyingModels = false
            }
        case "runtime_status":
            runtimeReady = event.payload["ready"]?.bool ?? false
            runtimeLabel = runtimeReady ? "FunASR 与 PyTorch 已安装" : "缺少 FunASR 或 PyTorch（开发版需在 .venv 安装）"
        case "download_progress":
            let key = event.payload["key"]?.string ?? ""
            let total = event.payload["size"]?.double ?? event.payload["total_size"]?.double ?? 1
            let downloaded = event.payload["downloaded"]?.double ?? 0
            let stage = event.payload["stage"]?.string ?? "downloading"
            let message = event.payload["message"]?.string ?? ""
            let speed = event.payload["speed"]?.double ?? 0
            let indeterminate = event.payload["indeterminate"]?.bool ?? false
            let source = event.payload["source"]?.string ?? ""
            let progress = total > 0 ? min(downloaded / total, 1.0) : 0
            if !key.isEmpty {
                downloadProgressByKey[key] = progress
                downloadStageByKey[key] = stage
                if !message.isEmpty {
                    downloadMessageByKey[key] = message
                }
                downloadSizeByKey[key] = Int64(total)
                downloadSpeedByKey[key] = speed
                downloadIndeterminateByKey[key] = indeterminate
                if !source.isEmpty {
                    downloadSourceByKey[key] = source
                }
                downloadingKey = key
            } else {
                downloadProgress = progress
            }
            downloadLabel = key.isEmpty ? (downloadLabel) : key
        case "model_install_status":
            if let status = event.payload["status"]?.string {
                if status == "ready" || status == "error" || status == "removed" || status == "cancelled" {
                    if let key = event.payload["key"]?.string {
                        downloadingKey = nil
                        downloadProgressByKey.removeValue(forKey: key)
                        downloadStageByKey.removeValue(forKey: key)
                        downloadMessageByKey.removeValue(forKey: key)
                        downloadSizeByKey.removeValue(forKey: key)
                        downloadSpeedByKey.removeValue(forKey: key)
                        downloadIndeterminateByKey.removeValue(forKey: key)
                        downloadSourceByKey.removeValue(forKey: key)
                    }
                    refreshModelCatalog()
                }
            }
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
        case "pending_voice_samples":
            pendingSamples = event.payload["samples"]?.array?.compactMap(VoiceSample.init) ?? []
        case "voice_samples_learned":
            isLearningSamples = false
            loadVoiceSamples()
            engine.send("get_status")
        case "voice_sample_added", "voice_sample_promoted", "voice_sample_deleted", "voice_profile_reset":
            loadVoiceSamples()
        case "voice_sample_confirmed", "voice_sample_rejected":
            loadVoiceSamples()
            loadPendingVoiceSamples()
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
        case "voice_enroll_status":
            let status = event.payload["status"]?.string ?? "failed"
            enrollStatus = status
            if let level = event.payload["level"]?.double {
                enrollLevel = level
            }
            if status == "done" || status == "failed" {
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
                    if self?.enrollStatus == status {
                        self?.enrollStatus = "idle"
                    }
                }
            }
        case "lexicon_list":
            lexiconWords = event.payload["words"]?.array?.compactMap(LexiconItem.init) ?? []
        case "lexicon_updated":
            loadLexicon()
            // Show toast if words were learned from corrections
            if let words = event.payload["words"]?.array?.compactMap({ $0.string }),
               !words.isEmpty,
               event.payload["source"]?.string == "correction" || event.payload["source"]?.string == "manual_learn" {
                learnedWordsToast = "已学习新词汇：" + words.joined(separator: "、")
                showLearnedWordsToast = true
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: 3_000_000_000)
                    if self.showLearnedWordsToast {
                        self.showLearnedWordsToast = false
                        self.learnedWordsToast = nil
                    }
                }
            }
        case "corrections_learned":
            loadLexicon()
            if let words = event.payload["words"]?.array?.compactMap({ $0.string }),
               !words.isEmpty {
                learnedWordsToast = "从纠错历史学习到 \(words.count) 个词：" + words.joined(separator: "、")
                showLearnedWordsToast = true
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: 3_000_000_000)
                    if self.showLearnedWordsToast {
                        self.showLearnedWordsToast = false
                        self.learnedWordsToast = nil
                    }
                }
            }
        case "error":
            if engineState == .loading || engineState == .stopping {
                engineState = .error
            }
            isLearningSamples = false
            errorMessage = event.payload["message"]?.string ?? "未知错误"
        default:
            break
        }
    }

    /// 环境音类别 → 中文显示名
    private func envDisplayName(_ category: String) -> String {
        switch category {
        case "typing": return "键盘打字"
        case "music": return "音乐"
        case "background_speech": return "背景人声"
        case "background_noise": return "环境噪声"
        case "alert": return "警报声"
        case "animal": return "动物声"
        case "impact": return "撞击声"
        case "appliance": return "家电运行"
        case "silence": return "安静"
        case "": return ""
        default: return category.replacingOccurrences(of: "_", with: " ")
        }
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

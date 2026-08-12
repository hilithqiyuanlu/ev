import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var model: AppModel
    @State private var showLogs = false
    @State private var showVoiceSamples = false
    @State private var showOnboarding = true

    private let modelNames = [
        "vad": "FSMN-VAD",
        "asr_streaming": "Paraformer Streaming",
        "asr_final": "Paraformer Large",
        "speaker": "ERes2NetV2",
    ]

    var body: some View {
        CenteredPage(maxWidth: 720) {
            VStack(alignment: .leading, spacing: 22) {
                onboardingSection
                Divider()
                modelsSection
                Divider()
                voiceProfileSection
                Divider()
                thresholdsSection
                Divider()
                storageSection
                Divider()
                systemSection
                Divider()
                logsSection
            }
        }
        .navigationTitle("设置")
        .onAppear {
            if model.isOnboardingComplete {
                showOnboarding = false
            }
        }
    }

    // MARK: - Onboarding

    private var onboardingSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button {
                withAnimation { showOnboarding.toggle() }
            } label: {
                HStack {
                    sectionHeader("准备检查", systemImage: "checklist.checked")
                    Spacer()
                    if model.isOnboardingComplete {
                        Image(systemName: showOnboarding ? "chevron.down" : "chevron.right")
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .buttonStyle(.plain)

            if !model.isOnboardingComplete || showOnboarding {
                if !model.isOnboardingComplete {
                    Text("按顺序完成以下步骤后即可开始使用 EV")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                VStack(spacing: 6) {
                    ChecklistRow("模型已下载并校验", done: model.onboardingChecks.models,
                                detail: model.onboardingChecks.models ? "四个模型均已就绪" : "请在下方下载并校验模型")
                    ChecklistRow("麦克风权限已允许", done: model.onboardingChecks.permission,
                                detail: permissionDetail)
                    ChecklistRow("声纹已建立", done: model.voiceProfile.isReady,
                                detail: voiceProfileDetail)
                }
            }
        }
    }

    private var permissionDetail: String {
        switch model.microphonePermission {
        case .authorized: return "已允许"
        case .denied, .restricted: return "已拒绝，请在系统设置中开启"
        case .notDetermined: return "首次监听时会请求权限"
        }
    }

    // MARK: - Models

    private var modelsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                sectionHeader("模型", systemImage: "cube.box")
                Spacer()
                if model.isVerifyingModels {
                    ProgressView().controlSize(.small)
                    Text("校验中...").font(.caption).foregroundStyle(.secondary)
                } else if model.showVerificationDone {
                    HStack(spacing: 4) {
                        statusIcon("checkmark.circle.fill", color: .green)
                        Text("校验完成")
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.green)
                    }
                }
            }
            Text(model.allModelsReady ? "四个模型均已就绪" : (model.models.isEmpty ? "正在检查模型..." : "部分模型尚未下载"))
                .font(.caption)
                .foregroundStyle(.secondary)

            if model.downloadProgress > 0 && model.downloadProgress < 1 {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: model.downloadProgress)
                    Text("正在下载 \(model.downloadLabel)… \(Int(model.downloadProgress * 100))%")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                ForEach(model.models) { ms in
                    HStack(spacing: 10) {
                        statusIcon(
                            ms.ready ? "checkmark.circle.fill" : (!ms.path.isEmpty ? "exclamationmark.triangle.fill" : "circle"),
                            color: ms.ready ? .green : (!ms.path.isEmpty ? .orange : .secondary)
                        )
                        Text(modelNames[ms.key] ?? ms.key)
                            .font(.subheadline)
                        Spacer()
                        if ms.path.isEmpty && model.downloadProgress == 0 {
                            Text("未下载").font(.caption).foregroundStyle(.secondary)
                        } else if !ms.ready {
                            Text("校验失败").font(.caption).foregroundStyle(.orange)
                        }
                    }
                }
            }

            HStack(spacing: 12) {
                if model.downloadProgress > 0 && model.downloadProgress < 1 {
                    Button(role: .destructive) { model.cancelDownload() } label: {
                        Label("取消下载", systemImage: "xmark.circle")
                    }
                } else {
                    Button { model.downloadModels() } label: {
                        Label("下载模型", systemImage: "arrow.down.circle")
                    }
                    .disabled(model.allModelsReady)
                }
                Button { model.verifyModels() } label: {
                    Label("重新校验", systemImage: "arrow.clockwise")
                }
                .disabled(model.isVerifyingModels || model.downloadProgress > 0)
            }
            .buttonStyle(.bordered)
        }
    }

    // MARK: - Voice Profile

    private var voiceProfileSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                sectionHeader("用户声纹", systemImage: "person.wave.2")
                Spacer()
                voiceStatusBadge
            }

            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 16) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("核心样本")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text("\(model.voiceProfile.coreCount) / 20")
                            .font(.title3.monospacedDigit())
                            .fontWeight(.medium)
                            .foregroundStyle(.orange)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text("缓存样本")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text("\(model.voiceProfile.cacheCount) / 50")
                            .font(.title3.monospacedDigit())
                            .fontWeight(.medium)
                            .foregroundStyle(.secondary)
                    }

                    if model.voiceProfile.centroidCount > 0 {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("质心数")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text("\(model.voiceProfile.centroidCount)")
                                .font(.title3.monospacedDigit())
                                .fontWeight(.medium)
                                .foregroundStyle(.blue)
                        }
                    }

                    Spacer()

                    VStack(alignment: .trailing, spacing: 2) {
                        Text("上次更新")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text(lastUpdatedText)
                            .font(.caption.monospacedDigit())
                    }
                }

                ProgressView(
                    value: Double(min(model.voiceProfile.coreCount, 20)),
                    total: 20
                )
                .opacity(model.voiceProfile.coreCount > 0 ? 1 : 0.3)
                .tint(.orange)
            }
            .padding(10)
            .background(Color.secondary.opacity(0.06))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            HStack(spacing: 10) {
                Toggle("自动学习", isOn: Binding(
                    get: { model.autoLearnEnabled },
                    set: { model.setAutoLearn($0) }
                ))
                .toggleStyle(.switch)
                .labelsHidden()
                Text("自动学习")
                    .font(.subheadline)
                Spacer()
                Menu {
                    Button("录制 2 秒") { model.manualEnrollDuration = 2.0; model.captureManualSample() }
                    Button("录制 3 秒") { model.manualEnrollDuration = 3.0; model.captureManualSample() }
                    Button("录制 5 秒") { model.manualEnrollDuration = 5.0; model.captureManualSample() }
                } label: {
                    Label(manualEnrollButtonText, systemImage: manualEnrollIcon)
                        .foregroundStyle(manualEnrollColor)
                }
                .disabled(model.manualEnrollStatus == "recording" || model.manualEnrollStatus == "processing")
                Button {
                    model.loadVoiceSamples()
                    showVoiceSamples = true
                } label: {
                    Label("管理样本", systemImage: "list.bullet.rectangle")
                }
            }

            if let error = model.manualEnrollError {
                Text("录入失败: \(error)")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .sheet(isPresented: $showVoiceSamples) {
            VoiceSamplesSheet()
                .environmentObject(model)
        }
    }

    private var manualEnrollButtonText: String {
        switch model.manualEnrollStatus {
        case "recording": return "录制中..."
        case "processing": return "处理中..."
        case "done": return "已添加"
        case "failed": return "失败，重试"
        default: return "手动添加样本"
        }
    }

    private var manualEnrollIcon: String {
        switch model.manualEnrollStatus {
        case "recording": return "record.circle"
        case "processing": return "gearshape.2"
        case "done": return "checkmark.circle"
        case "failed": return "exclamationmark.triangle"
        default: return "plus.circle"
        }
    }

    private var manualEnrollColor: Color {
        switch model.manualEnrollStatus {
        case "recording": return .red
        case "done": return .green
        case "failed": return .orange
        default: return .accentColor
        }
    }

    private var voiceStatusBadge: some View {
        let (text, color, icon): (String, Color, String) = {
            if model.voiceProfile.isReady {
                return ("已就绪", .green, "checkmark.circle.fill")
            } else if model.voiceProfile.sampleCount > 0 {
                return ("学习中", .orange, "circle.dotted")
            } else {
                return ("未建立", .secondary, "circle")
            }
        }()
        return HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 12))
            Text(text)
        }
        .font(.caption.weight(.medium))
        .foregroundStyle(color)
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(color.opacity(0.12))
        .clipShape(Capsule())
    }

    private var lastUpdatedText: String {
        guard let dateStr = model.voiceProfile.lastUpdated else { return "—" }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: dateStr) {
            let rel = RelativeDateTimeFormatter()
            rel.unitsStyle = .abbreviated
            rel.locale = Locale(identifier: "zh_CN")
            return rel.localizedString(for: date, relativeTo: Date())
        }
        return String(dateStr.prefix(10))
    }

    private var voiceProfileDetail: String {
        if model.voiceProfile.sampleCount == 0 {
            return "正常使用时自动收集"
        }
        if model.voiceProfile.isReady {
            return "核心 \(model.voiceProfile.coreCount)、缓存 \(model.voiceProfile.cacheCount)，\(model.voiceProfile.centroidCount) 个质心"
        }
        return "学习中（核心 \(model.voiceProfile.coreCount)/3）"
    }

    // MARK: - Thresholds

    private var thresholdsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("声纹判定阈值", systemImage: "slider.horizontal.3")
            Text("高于此分数判定为本人，低于则判定为他人。声纹建立前（<3个核心样本）所有语音均接收。")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Text("阈值")
                Slider(value: $model.speakerThreshold, in: 0.3...0.8, onEditingChanged: { editing in
                    if !editing { model.saveThresholds() }
                })
                Text(String(format: "%.2f", model.speakerThreshold)).monospacedDigit()
            }
            .font(.subheadline)
        }
    }

    // MARK: - Storage

    private var storageSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("存储", systemImage: "folder.fill")
            HStack(spacing: 10) {
                Text("数据目录:").font(.subheadline)
                Text(model.applicationSupportPath)
                    .font(.caption).foregroundStyle(.secondary)
                    .textSelection(.enabled)
                Button("打开") { model.openApplicationSupport() }
            }
            HStack(spacing: 10) {
                Text("日志目录:").font(.subheadline)
                Button("打开日志") { model.openLogs() }
            }
        }
    }

    // MARK: - System

    private var systemSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("系统", systemImage: "desktopcomputer")
            Toggle("登录时启动 EV", isOn: Binding(
                get: { model.launchAtLogin },
                set: { model.setLaunchAtLogin($0) }
            ))
            Toggle("Ctrl+T 启用/停止监听", isOn: Binding(
                get: { model.ctrlTToggleListening },
                set: { model.setCtrlTToggleListening($0) }
            ))
        }
    }

    // MARK: - Logs

    private var logsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button {
                withAnimation { showLogs.toggle() }
            } label: {
                HStack {
                    sectionHeader("引擎日志", systemImage: "doc.text.fill")
                    Spacer()
                    Image(systemName: showLogs ? "chevron.down" : "chevron.right")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)
            if showLogs {
                ScrollView {
                    Text(model.lastEngineLog.isEmpty ? "（暂无日志）" : model.lastEngineLog)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(height: 140)
                .padding(8)
                .background(Color.secondary.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 4))
            }
        }
    }

    // MARK: - Shared helpers

    private func sectionHeader(_ title: String, systemImage: String) -> some View {
        Label(title, systemImage: systemImage)
            .font(.headline)
    }

    private func statusIcon(_ name: String, color: Color) -> some View {
        Image(systemName: name)
            .font(.system(size: 16))
            .foregroundStyle(color)
            .frame(width: 18, alignment: .center)
    }
}

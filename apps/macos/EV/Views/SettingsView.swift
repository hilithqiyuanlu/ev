import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var model: AppModel
    @State private var showLogs = false
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
                    ChecklistRow("声纹已建立", done: !model.needsVoiceOnboarding,
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

    private var voiceProfileDetail: String {
        if model.needsVoiceOnboarding {
            return model.voiceProfile.coreCount == 0
                ? "尚未录入，请在「声纹」页完成引导"
                : "已录 \(model.onboardingCount)/\(model.onboardingTarget) 段"
        }
        return "已建立（核心 \(model.voiceProfile.coreCount)、缓存 \(model.voiceProfile.cacheCount)、\(model.voiceProfile.centroidCount) 质心）"
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

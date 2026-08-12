import SwiftUI

struct VoiceView: View {
    @EnvironmentObject private var model: AppModel
    @State private var playingSampleId: String?
    @State private var showResetConfirm = false

    private var coreSamples: [VoiceSample] { model.voiceSamples.filter { $0.tier == "core" } }
    private var cacheSamples: [VoiceSample] { model.voiceSamples.filter { $0.tier == "cache" } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                fingerprintCard
                if !model.pendingSamples.isEmpty {
                    pendingSection
                }
                sampleList
                thresholdSection
            }
            .padding(24)
            .frame(maxWidth: 760)
            .frame(maxWidth: .infinity)
        }
        .navigationTitle("声纹")
        .onAppear {
            model.loadVoiceSamples()
            model.loadPendingVoiceSamples()
        }
        .alert("重置声纹", isPresented: $showResetConfirm) {
            Button("取消", role: .cancel) {}
            Button("重置", role: .destructive) {
                model.resetVoiceProfile()
                playingSampleId = nil
            }
        } message: {
            Text("将删除全部 \(model.voiceSamples.count) 个声纹样本，回到引导录制状态。")
        }
    }

    // MARK: - Fingerprint card

    private var fingerprintCard: some View {
        let needsOnboarding = model.needsVoiceOnboarding
        return HStack(alignment: .center, spacing: 20) {
            FingerprintView(
                progress: model.onboardingProgress,
                countText: needsOnboarding && model.onboardingCount > 0
                    ? "\(model.onboardingCount)/\(model.onboardingTarget)"
                    : nil,
                greyed: needsOnboarding && model.voiceProfile.coreCount == 0
            )

            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 10) {
                    Text("我的声纹")
                        .font(.title2.bold())
                    voiceStatusBadge
                }

                Text(descriptorText)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                if needsOnboarding {
                    HStack(spacing: 12) {
                        Text("已录入 \(model.onboardingCount) / \(model.onboardingTarget) 段")
                            .font(.subheadline.monospacedDigit().weight(.semibold))
                            .fixedSize()
                        ProgressView(value: model.onboardingProgress)
                            .progressViewStyle(.linear)
                            .frame(maxWidth: 180)
                    }
                    if model.voiceProfile.coreCount == 0 {
                        Text("安静环境下用平时音量说话，每段 2–10 秒即可")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    HStack(spacing: 14) {
                        Text("共 \(model.voiceProfile.sampleCount) 个样本")
                            .font(.subheadline.monospacedDigit())
                            .foregroundStyle(.secondary)
                        if !model.autoLearnEnabled {
                            Text("自动学习已关闭")
                                .font(.caption.weight(.medium))
                                .foregroundStyle(.orange)
                        }
                    }
                }
            }

            Spacer(minLength: 8)

            if needsOnboarding {
                enrollControl
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardStyle()
    }

    private var enrollControl: some View {
        HStack(spacing: 10) {
            if model.enrollStatus == "recording" {
                HStack(spacing: 10) {
                    enrollWaveform
                    Button {
                        model.stopVoiceEnrollment()
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "stop.fill")
                                .font(.system(size: 11, weight: .semibold))
                            Text("完成本段")
                                .font(.subheadline.weight(.medium))
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(CapsuleButtonStyle(isDestructive: true))
                }
            } else if model.enrollStatus == "processing" {
                ProgressView().controlSize(.small)
                Text("正在处理…").font(.caption).foregroundStyle(.secondary)
            } else {
                Button {
                    model.startVoiceEnrollment()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: model.voiceProfile.coreCount == 0 ? "waveform" : "plus")
                            .font(.system(size: 11, weight: .semibold))
                        Text(model.voiceProfile.coreCount == 0 ? "开始录入" : "继续录入")
                            .font(.subheadline.weight(.medium))
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .contentShape(Rectangle())
                }
                .buttonStyle(CapsuleButtonStyle())
                .disabled(model.enrollStatus == "processing")
            }

            if model.enrollStatus == "done" {
                Label("已添加，继续下一段", systemImage: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else if model.enrollStatus == "failed" {
                Button {
                    model.startVoiceEnrollment()
                } label: {
                    Text("录入失败，重试")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var enrollWaveform: some View {
        HStack(alignment: .center, spacing: 3) {
            ForEach(0..<5, id: \.self) { index in
                let base = max(0.05, model.enrollLevel + Double(index - 2) * 0.15)
                let height = max(3, min(24, base * 30))
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.red.opacity(0.7))
                    .frame(width: 4, height: height)
                    .animation(.easeOut(duration: 0.1), value: model.enrollLevel)
            }
        }
        .frame(width: 32, height: 28)
    }

    private var voiceStatusBadge: some View {
        let (text, color): (String, Color) = {
            if model.voiceProfile.coreCount >= model.onboardingTarget {
                return ("已就绪", .green)
            } else if model.voiceProfile.coreCount > 0 {
                return ("学习中", .orange)
            } else {
                return ("未建立", .secondary)
            }
        }()
        return Text(text)
            .font(.caption.weight(.medium))
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(color.opacity(0.12))
            .clipShape(Capsule())
    }

    private var descriptorText: String {
        if model.voiceProfile.coreCount == 0 {
            return "录制 \(model.onboardingTarget) 段语音，建立你的专属声纹"
        }
        if model.voiceProfile.coreCount >= model.onboardingTarget {
            return "已就绪，高置信度语音会持续优化声纹"
        }
        return "继续录制以完成引导"
    }

    // MARK: - Pending samples

    private var pendingSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Text("待确认样本")
                    .font(.headline)
                Text("\(model.pendingSamples.count)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Text("与当前声纹差异较大的样本，确认后纳入参考，否则删除。")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(model.pendingSamples) { sample in
                pendingRow(sample)
            }
        }
    }

    private func pendingRow(_ sample: VoiceSample) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "questionmark.circle")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.orange)
                .frame(width: 32, height: 32)
                .background(Circle().fill(Color.orange.opacity(0.1)))

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text(formatDuration(sample.durationMS))
                        .font(.subheadline.monospacedDigit())
                    Text("置信度 \(String(format: "%.2f", sample.score))")
                        .font(.caption)
                        .foregroundStyle(scoreColor(score: sample.score))
                }
                Text(formatDate(sample.createdAt))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 8)

            Button("确认") {
                model.confirmPendingVoiceSample(sample.id)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            Button("删除", role: .destructive) {
                model.rejectPendingVoiceSample(sample.id)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .cardStyle()
    }

    // MARK: - Sample list

    private var sampleList: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("样本列表")
                    .font(.headline)
                Spacer()
                if !model.voiceSamples.isEmpty {
                    HStack(spacing: 8) {
                        Button {
                            model.learnVoiceSamples()
                        } label: {
                            HStack(spacing: 6) {
                                if model.isLearningSamples {
                                    ProgressView().controlSize(.small)
                                } else {
                                    Image(systemName: "arrow.triangle.2.circlepath")
                                        .font(.system(size: 11, weight: .semibold))
                                }
                                Text("重新学习")
                                    .font(.subheadline.weight(.medium))
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(CapsuleButtonStyle())
                        .disabled(model.isLearningSamples)

                        Button(role: .destructive) {
                            showResetConfirm = true
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: "arrow.counterclockwise")
                                    .font(.system(size: 11, weight: .semibold))
                                Text("重置声纹")
                                    .font(.subheadline.weight(.medium))
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(CapsuleButtonStyle(isDestructive: true))
                        .disabled(model.voiceSamples.isEmpty)
                    }
                }
            }

            if model.voiceSamples.isEmpty {
                emptySampleState
            } else {
                if !coreSamples.isEmpty {
                    tierHeader(
                        title: "核心样本",
                        count: coreSamples.count,
                        subtitle: "高置信度样本，声纹判定的主要参考"
                    )
                    ForEach(Array(coreSamples.enumerated()), id: \.element.id) { index, sample in
                        sampleRow(sample, index: index, tier: "core")
                    }
                }
                if !cacheSamples.isEmpty {
                    tierHeader(
                        title: "缓存样本",
                        count: cacheSamples.count,
                        subtitle: "低置信度候选样本，可手动晋升为核心"
                    )
                    ForEach(Array(cacheSamples.enumerated()), id: \.element.id) { index, sample in
                        sampleRow(sample, index: index, tier: "cache")
                    }
                }
            }
        }
    }

    private var emptySampleState: some View {
        EmptyStateView("暂无样本，完成上方引导或正常使用自动收集", systemImage: "waveform")
            .frame(maxWidth: .infinity, minHeight: 160, alignment: .center)
    }

    private func tierHeader(title: String, count: Int, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Text("\(count)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Text(subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.top, 8)
        .padding(.bottom, 2)
    }

    private func sampleRow(_ sample: VoiceSample, index: Int, tier: String) -> some View {
        let isPlaying = playingSampleId == sample.id
        return HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(scoreColor(score: sample.score).opacity(0.15))
                    .frame(width: 32, height: 32)
                if sample.isManual {
                    Image(systemName: "hand.tap.fill")
                        .font(.caption2)
                        .foregroundStyle(scoreColor(score: sample.score))
                } else {
                    Text("\(index + 1)")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(scoreColor(score: sample.score))
                }
            }

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text(formatDuration(sample.durationMS))
                        .font(.subheadline.monospacedDigit())
                    if let hint = sample.transcriptHint, !hint.isEmpty {
                        Text(hint)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    if !sample.audioAvailable {
                        Label("音频缺失", systemImage: "exclamationmark.triangle")
                            .font(.caption2)
                            .foregroundStyle(.red)
                    }
                }
                HStack(spacing: 8) {
                    Text(formatDate(sample.createdAt))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("置信度 \(String(format: "%.2f", sample.score))")
                        .font(.caption)
                        .foregroundStyle(scoreColor(score: sample.score))
                }
            }

            Spacer(minLength: 8)

            if tier == "cache" {
                Button {
                    model.promoteVoiceSample(sample.id)
                } label: {
                    Image(systemName: "arrow.up.circle")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.borderless)
                .help("晋升为核心样本")
            }

            Button {
                guard sample.audioAvailable else { return }
                togglePlay(sample)
            } label: {
                Image(systemName: isPlaying ? "stop.fill" : "play.fill")
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.borderless)
            .disabled(!sample.audioAvailable)

            Button(role: .destructive) {
                model.deleteVoiceSample(sample.id)
            } label: {
                Image(systemName: "trash")
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.borderless)
            .help("删除样本")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(isPlaying ? Color.accentColor.opacity(0.06) : Color.primary.opacity(0.03))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(isPlaying ? Color.accentColor.opacity(0.15) : Color.primary.opacity(0.04), lineWidth: 0.5)
                )
        )
        .contentShape(Rectangle())
    }

    private func togglePlay(_ sample: VoiceSample) {
        if playingSampleId == sample.id {
            model.audioPlayer.stop()
            playingSampleId = nil
        } else {
            model.audioPlayer.stop()
            playingSampleId = sample.id
            model.audioPlayer.play(url: URL(fileURLWithPath: sample.audioPath)) {
                Task { @MainActor in
                    if playingSampleId == sample.id { playingSampleId = nil }
                }
            }
        }
    }

    // MARK: - Threshold

    private var thresholdSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("声纹判定阈值")
                .font(.headline)
            Text("判断语音是否来自你本人的敏感度：数值越低越宽松，越高越严格。")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack(spacing: 14) {
                Slider(value: $model.speakerThreshold, in: 0.3...0.8, onEditingChanged: { editing in
                    if !editing { model.saveThresholds() }
                })
                Text(String(format: "%.2f", model.speakerThreshold))
                    .font(.subheadline.monospacedDigit())
            }
        }
        .padding(16)
        .cardStyle()
    }

    // MARK: - Helpers

    private func scoreColor(score: Double) -> Color {
        if score >= 0.8 { return .green }
        if score >= 0.6 { return .orange }
        return .secondary
    }

    private func formatDuration(_ ms: Int) -> String {
        let seconds = Double(ms) / 1000.0
        if seconds < 60 { return String(format: "%.1fs", seconds) }
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return "\(mins)m\(secs)s"
    }

    private func formatDate(_ iso: String) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        guard let date = formatter.date(from: iso) else { return iso }
        let rel = RelativeDateTimeFormatter()
        rel.unitsStyle = .abbreviated
        rel.locale = Locale(identifier: "zh_CN")
        return rel.localizedString(for: date, relativeTo: Date())
    }
}

// MARK: - Card style

private struct CardStyle: ViewModifier {
    var fill: Color = Color.primary.opacity(0.03)

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(fill)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.primary.opacity(0.04), lineWidth: 0.5)
                    )
            )
    }
}

private extension View {
    func cardStyle(fill: Color = Color.primary.opacity(0.03)) -> some View {
        modifier(CardStyle(fill: fill))
    }
}

// MARK: - Fingerprint visual

struct FingerprintView: View {
    let progress: Double
    let countText: String?
    let greyed: Bool

    private var fillFraction: Double {
        guard !greyed else { return 0 }
        return max(0, min(progress, 1))
    }

    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.secondary.opacity(0.12), lineWidth: 5)
            Circle()
                .trim(from: 0, to: fillFraction)
                .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 5, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.easeOut(duration: 0.3), value: fillFraction)
            centerContent
        }
        .frame(width: 84, height: 84)
        .padding(14)
        .background(
            Circle().fill(Color.primary.opacity(0.03))
        )
        .overlay(
            Circle().stroke(Color.primary.opacity(0.04), lineWidth: 0.5)
        )
    }

    @ViewBuilder
    private var centerContent: some View {
        if greyed {
            Image(systemName: "waveform")
                .font(.system(size: 28, weight: .medium))
                .foregroundStyle(.tertiary)
        } else if let countText {
            Text(countText)
                .font(.system(.subheadline, design: .rounded, weight: .semibold))
                .monospacedDigit()
                .foregroundStyle(.primary)
        } else {
            Image(systemName: "person.wave.2")
                .font(.system(size: 30, weight: .medium))
                .foregroundStyle(Color.accentColor)
        }
    }
}

#Preview {
    VoiceView()
        .environmentObject(AppModel(engine: MockEngineTransportPreview()))
}

private class MockEngineTransportPreview: EngineTransport {
    var onEvent: (@Sendable (EngineEnvelope) -> Void)?
    var onTermination: (@Sendable (Int32) -> Void)?
    var onStderr: (@Sendable (String) -> Void)?
    var isRunning: Bool { false }
    func start() throws {}
    func send(_ command: String, payload: [String: Any]) {}
    func shutdown() {}
}

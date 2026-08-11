import SwiftUI

struct VoiceView: View {
    @EnvironmentObject private var model: AppModel
    @State private var playingSampleId: String?
    @State private var showResetConfirm = false

    private var coreSamples: [VoiceSample] { model.voiceSamples.filter { $0.tier == "core" } }
    private var cacheSamples: [VoiceSample] { model.voiceSamples.filter { $0.tier == "cache" } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                fingerprintCard
                Divider()
                if !model.pendingSamples.isEmpty {
                    pendingSection
                    Divider()
                }
                sampleList
                Divider()
                thresholdSection
            }
            .padding(20)
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
        return HStack(alignment: .top, spacing: 22) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    Text("我的声纹")
                        .font(.title2.bold())
                    voiceStatusBadge
                }

                Text(descriptorText)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                if needsOnboarding {
                    Text("已录入 \(model.onboardingCount) / \(model.onboardingTarget) 段")
                        .font(.subheadline.monospacedDigit().weight(.semibold))
                        .foregroundStyle(Color.accentColor)
                }

                HStack(spacing: 20) {
                    statItem("核心", value: "\(model.voiceProfile.coreCount)/\(model.voiceProfile.coreCount + model.voiceProfile.cacheCount)", color: .orange)
                    statItem("缓存", value: "\(model.voiceProfile.cacheCount)", color: .secondary)
                    if model.voiceProfile.centroidCount > 0 {
                        statItem("质心", value: "\(model.voiceProfile.centroidCount)", color: .blue)
                    }
                }

                if needsOnboarding {
                    enrollControl
                }
            }

            Spacer()

            FingerprintView(
                progress: model.onboardingProgress,
                greyed: needsOnboarding && model.voiceProfile.coreCount == 0
            )
            .frame(width: 112, height: 136)
        }
        .padding(18)
        .background(Color.secondary.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var enrollControl: some View {
        HStack(spacing: 10) {
            if model.enrollStatus == "recording" {
                HStack(spacing: 10) {
                    enrollWaveform
                    Button {
                        model.stopVoiceEnrollment()
                    } label: {
                        Text("完成本段")
                            .fontWeight(.medium)
                    }
                    .tint(.red)
                    .buttonStyle(.borderedProminent)
                }
            } else if model.enrollStatus == "processing" {
                ProgressView().controlSize(.small)
                Text("正在处理…").font(.caption).foregroundStyle(.secondary)
            } else {
                Button {
                    model.startVoiceEnrollment()
                } label: {
                    Text(model.voiceProfile.coreCount == 0 ? "开始录入" : "继续录入")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.enrollStatus == "processing")
            }

            if model.enrollStatus == "done" {
                Text("已添加，继续下一段")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else if model.enrollStatus == "failed" {
                Button {
                    model.startVoiceEnrollment()
                } label: {
                    Text("录入失败，重试")
                        .font(.caption)
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
            return "尚未录制，完成引导后自动学习开启"
        }
        if model.voiceProfile.coreCount >= model.onboardingTarget {
            return "已就绪，高置信度语音会持续优化声纹"
        }
        return "继续录制以完成引导"
    }

    private func statItem(_ label: String, value: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.monospacedDigit().weight(.semibold))
                .foregroundStyle(color)
        }
    }

    // MARK: - Sample list

    private var pendingSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("待确认样本")
                    .font(.headline)
                Spacer()
                Text("与声纹距离较大的缓存样本，确认后晋升核心")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ForEach(model.pendingSamples) { sample in
                HStack(spacing: 10) {
                    Image(systemName: "questionmark.circle")
                        .foregroundStyle(.orange)
                    Text(URL(fileURLWithPath: sample.audioPath).lastPathComponent)
                        .lineLimit(1)
                    Spacer()
                    Text("置信度 \(String(format: "%.2f", sample.score))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
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
                .padding(.vertical, 6)
            }
        }
    }

    private var sampleList: some View {
        VStack(alignment: .leading, spacing: 8) {
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
                VStack(spacing: 8) {
                    Image(systemName: "waveform")
                        .font(.system(size: 34))
                        .foregroundStyle(.tertiary)
                    Text("暂无样本")
                        .font(.headline)
                    Text("完成上方引导录制，或正常使用 EV 自动收集")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 28)
            } else {
                if !coreSamples.isEmpty {
                    tierHeader(title: "核心样本", systemImage: "star.fill", color: .orange, count: coreSamples.count)
                    ForEach(Array(coreSamples.enumerated()), id: \.element.id) { index, sample in
                        sampleRow(sample, index: index, tier: "core")
                    }
                }
                if !cacheSamples.isEmpty {
                    tierHeader(title: "缓存样本", systemImage: "tray", color: .secondary, count: cacheSamples.count)
                    ForEach(Array(cacheSamples.enumerated()), id: \.element.id) { index, sample in
                        sampleRow(sample, index: index, tier: "cache")
                    }
                }
            }
        }
    }

    private func tierHeader(title: String, systemImage: String, color: Color, count: Int) -> some View {
        HStack(spacing: 8) {
            Image(systemName: systemImage)
                .foregroundStyle(color)
            Text(title)
                .font(.subheadline.weight(.semibold))
            Text("(\(count))")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.vertical, 6)
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

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(formatDuration(sample.durationMS))
                        .font(.subheadline.monospacedDigit())
                    Text("·")
                        .foregroundStyle(.tertiary)
                    Text("置信度 \(String(format: "%.2f", sample.score))")
                        .font(.caption)
                        .foregroundStyle(scoreColor(score: sample.score))
                    if !sample.audioAvailable {
                        Text("音频缺失")
                            .font(.caption2)
                            .foregroundStyle(.red)
                    }
                }
                Text(formatDate(sample.createdAt))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if tier == "cache" {
                Button {
                    model.promoteVoiceSample(sample.id)
                } label: {
                    Image(systemName: "arrow.up.circle")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.borderless)
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
        }
        .padding(.vertical, 8)
        .background(isPlaying ? Color.accentColor.opacity(0.06) : Color.clear)
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
        VStack(alignment: .leading, spacing: 8) {
            Text("声纹判定阈值")
                .font(.headline)
            HStack(spacing: 12) {
                Slider(value: $model.speakerThreshold, in: 0.3...0.8, onEditingChanged: { editing in
                    if !editing { model.saveThresholds() }
                })
                Text(String(format: "%.2f", model.speakerThreshold)).monospacedDigit()
            }
            .font(.subheadline)
        }
        .padding(12)
        .background(Color.secondary.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 10))
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

// MARK: - Fingerprint visual

struct FingerprintView: View {
    let progress: Double
    let greyed: Bool

    private let ridges = 5

    private var filledRidges: Int {
        guard !greyed else { return 0 }
        return max(0, min(ridges, Int((progress * Double(ridges)).rounded(.up))))
    }

    var body: some View {
        Canvas { context, size in
            let w = size.width
            let h = size.height
            let bottom = h * 0.98
            for i in 0..<ridges {
                let t = Double(i) / Double(ridges - 1)
                let archW = w * (0.42 + 0.34 * t)
                let archH = h * (0.18 + 0.62 * t)
                let top = h * (0.94 - archH)
                let x0 = (w - archW) / 2
                var path = Path()
                path.move(to: CGPoint(x: x0, y: bottom))
                path.addQuadCurve(
                    to: CGPoint(x: x0 + archW, y: bottom),
                    control: CGPoint(x: x0 + archW / 2, y: top)
                )
                let isFilled = i < filledRidges
                let color: Color = isFilled
                    ? .accentColor
                    : Color.secondary.opacity(greyed ? 0.12 : 0.28)
                context.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 3.4, lineCap: .round))
            }
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
import SwiftUI

struct VoiceView: View {
    @EnvironmentObject private var model: AppModel
    @State private var playingSampleId: String?
    @State private var showResetConfirm = false

    private var coreSamples: [VoiceSample] { model.voiceSamples.filter { $0.tier == "core" } }
    private var cacheSamples: [VoiceSample] { model.voiceSamples.filter { $0.tier == "cache" } }
    private var learningInProgress: Bool { model.isLearningSamples }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                fingerprintCard
                actionBar
                Divider()
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
        }
        .alert("重置声纹", isPresented: $showResetConfirm) {
            Button("取消", role: .cancel) {}
            Button("重置", role: .destructive) {
                model.resetVoiceProfile()
                playingSampleId = nil
            }
        } message: {
            Text("将删除全部 \(model.voiceSamples.count) 个声纹样本，回到引导录制状态。此操作不可撤销。")
        }
    }

    // MARK: - Fingerprint card

    private var fingerprintCard: some View {
        let needsOnboarding = model.needsVoiceOnboarding
        return VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center, spacing: 20) {
                FingerprintView(
                    progress: model.onboardingProgress,
                    centroidCount: model.voiceProfile.centroidCount,
                    greyed: needsOnboarding && model.voiceProfile.coreCount == 0
                )
                .frame(width: 92, height: 104)

                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        Text("我的声纹")
                            .font(.title3.bold())
                        voiceStatusBadge
                    }

                    HStack(spacing: 16) {
                        statItem("核心", value: "\(model.voiceProfile.coreCount)/20", color: .orange)
                        statItem("缓存", value: "\(model.voiceProfile.cacheCount)/50", color: .secondary)
                        if model.voiceProfile.centroidCount > 0 {
                            statItem("质心", value: "\(model.voiceProfile.centroidCount)", color: .blue)
                        }
                    }

                    Text(descriptorText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(16)
            .background(Color.secondary.opacity(0.06))
            .clipShape(RoundedRectangle(cornerRadius: 12))

            if needsOnboarding {
                onboardingGuide
            } else {
                HStack(spacing: 6) {
                    Image(systemName: "waveform.and.mic")
                        .foregroundStyle(.green)
                    Text(autoLearnText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var onboardingGuide: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(model.voiceProfile.coreCount == 0
                ? "先录入你的声音，建立专属声纹"
                : "录入第 \(model.onboardingCount + 1) / \(model.onboardingTarget) 段")
                .font(.headline)
            Text("完成 \(model.onboardingTarget) 段录音后，声纹会自动学习，日常语音会不断优化它。")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("提示：在安静环境、用正常音量和语速，每段说 3–5 秒的完整句子（不必刻意大声或小声）。")
                .font(.caption)
                .foregroundStyle(.tertiary)

            enrollControl
        }
        .padding(12)
        .background(Color.accentColor.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 10))
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
                            Image(systemName: "stop.circle.fill")
                            Text("完成本段")
                                .fontWeight(.medium)
                        }
                        .foregroundStyle(.white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 6)
                        .background(Color.red, in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
            } else if model.enrollStatus == "processing" {
                ProgressView().controlSize(.small)
                Text("正在处理…").font(.caption).foregroundStyle(.secondary)
            } else {
                Button {
                    model.startVoiceEnrollment()
                } label: {
                    Label(model.voiceProfile.coreCount == 0 ? "开始录入" : "继续录入",
                          systemImage: "waveform.badge.mic")
                }
                .tint(.accentColor)
                .disabled(model.enrollStatus == "processing")
            }

            if model.enrollStatus == "done" {
                Label("已添加，继续下一段", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            } else if model.enrollStatus == "failed" {
                Label("录入失败，重试", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
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
        let (text, color, icon): (String, Color, String) = {
            if model.voiceProfile.coreCount >= model.onboardingTarget {
                return ("已就绪", .green, "checkmark.circle.fill")
            } else if model.voiceProfile.coreCount > 0 {
                return ("学习中 \(model.onboardingCount)/\(model.onboardingTarget)", .orange, "circle.dotted")
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

    private var descriptorText: String {
        if model.voiceProfile.coreCount == 0 {
            return "尚无样本，完成引导后会开启自动学习"
        }
        if model.voiceProfile.coreCount >= model.onboardingTarget {
            return "自动学习中 · 高置信度语音会不断优化声纹"
        }
        return "已完成 \(model.onboardingCount) / \(model.onboardingTarget) 段录音"
    }

    private var autoLearnText: String {
        "自动学习中：新语音每段会按置信度更新声纹（最近更新 \(lastUpdatedText)）"
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

    // MARK: - Actions

    private var actionBar: some View {
        HStack(spacing: 10) {
            Button {
                model.learnVoiceSamples()
            } label: {
                if learningInProgress {
                    ProgressView().controlSize(.small)
                }
                Label("学习样本", systemImage: "arrow.triangle.2.circlepath")
            }
            .tint(.accentColor)
            .disabled(learningInProgress || model.voiceSamples.isEmpty)
            .help("用当前样本的音频重新过一遍声纹模型并重建质心（可修复因模型更新导致的陈旧特征）")

            Button(role: .destructive) {
                showResetConfirm = true
            } label: {
                Label("重置样本", systemImage: "arrow.counterclockwise")
            }
            .disabled(model.voiceSamples.isEmpty)

            Spacer()

            if model.enrollStatus == "recording" {
                Label("正在录音…", systemImage: "record.circle")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
    }

    // MARK: - Sample list

    private var sampleList: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("样本列表")
                .font(.headline)
            Text("核心样本用于建模；缓存样本用于观察与提升。缺失音频的样本请在删除后重新学习。")
                .font(.caption)
                .foregroundStyle(.secondary)

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
                    tierHeader(title: "核心样本", subtitle: "用于建模，最多20个，优先保留高质量样本",
                               systemImage: "star.fill", color: .orange, count: coreSamples.count)
                    ForEach(Array(coreSamples.enumerated()), id: \.element.id) { index, sample in
                        sampleRow(sample, index: index, tier: "core")
                    }
                }
                if !cacheSamples.isEmpty {
                    tierHeader(title: "缓存样本", subtitle: "仅记录不用于建模，最多50条",
                               systemImage: "tray", color: .secondary, count: cacheSamples.count)
                    ForEach(Array(cacheSamples.enumerated()), id: \.element.id) { index, sample in
                        sampleRow(sample, index: index, tier: "cache")
                    }
                }
            }
        }
    }

    private func tierHeader(title: String, subtitle: String, systemImage: String, color: Color, count: Int) -> some View {
        HStack(spacing: 8) {
            Image(systemName: systemImage)
                .foregroundStyle(color)
            Text(title)
                .font(.subheadline.weight(.semibold))
            Text("(\(count))")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Text(subtitle)
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 6)
    }

    private func sampleRow(_ sample: VoiceSample, index: Int, tier: String) -> some View {
        let isPlaying = playingSampleId == sample.id
        let isCore = tier == "core"
        return HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(scoreColor(score: sample.score).opacity(isCore ? 0.15 : 0.08))
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
                    if sample.isManual {
                        Text("手动")
                            .font(.caption2)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(Color.blue.opacity(0.1))
                            .foregroundStyle(.blue)
                            .clipShape(RoundedRectangle(cornerRadius: 3))
                    }
                    if !sample.audioAvailable {
                        Text("音频缺失")
                            .font(.caption2)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(Color.red.opacity(0.1))
                            .foregroundStyle(.red)
                            .clipShape(RoundedRectangle(cornerRadius: 3))
                    }
                }
                Text(formatDate(sample.createdAt))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if !isCore {
                Button {
                    model.promoteVoiceSample(sample.id)
                } label: {
                    Image(systemName: "arrow.up.circle")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.borderless)
                .help("提升到核心样本")
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
            Label("声纹判定阈值", systemImage: "slider.horizontal.3")
                .font(.headline)
            Text("高于此分数判定为本人语音并用于自动学习，低于则忽略。默认 0.40。")
                .font(.caption)
                .foregroundStyle(.secondary)
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

    private var lastUpdatedText: String {
        guard let dateStr = model.voiceProfile.lastUpdated else { return "—" }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        guard let date = formatter.date(from: dateStr) else { return String(dateStr.prefix(10)) }
        let rel = RelativeDateTimeFormatter()
        rel.unitsStyle = .abbreviated
        rel.locale = Locale(identifier: "zh_CN")
        return rel.localizedString(for: date, relativeTo: Date())
    }
}

// MARK: - Fingerprint visual

struct FingerprintView: View {
    let progress: Double
    let centroidCount: Int
    let greyed: Bool

    var body: some View {
        Canvas { context, size in
            let rings = 7
            let filled = min(rings, max(0, Int((progress * Double(rings)).rounded(.up))))
            for i in 0..<rings {
                let t = Double(i) / Double(rings - 1)
                let rx = size.width * (0.16 + 0.34 * t)
                let ry = size.height * (0.42 + 0.58 * t)
                let rect = CGRect(
                    x: (size.width - rx) / 2,
                    y: size.height - ry * 1.05,
                    width: rx,
                    height: ry * 1.05
                )
                var path = Path(ellipseIn: rect)
                let isFilled = i < filled && !greyed
                let color: Color = isFilled
                    ? .accentColor
                    : Color.secondary.opacity(greyed ? 0.14 : 0.30)
                context.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 3.2, lineCap: .round))
            }
        }
        .clipShape(ArchedClip())
        .overlay {
            if greyed {
                Image(systemName: "plus.magnifyingglass")
                    .font(.system(size: 22, weight: .medium))
                    .foregroundStyle(.secondary.opacity(0.8))
            } else {
                VStack(spacing: 4) {
                    Spacer()
                    HStack(spacing: 5) {
                        ForEach(0..<3, id: \.self) { i in
                            Circle()
                                .stroke(i < centroidCount ? Color.blue : Color.secondary.opacity(0.3),
                                        lineWidth: 2)
                                .frame(width: 8, height: 8)
                        }
                    }
                    .padding(.bottom, 6)
                }
            }
        }
    }
}

struct ArchedClip: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: rect.minX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY * 0.62))
        p.addQuadCurve(
            to: CGPoint(x: rect.maxX, y: rect.maxY * 0.62),
            control: CGPoint(x: rect.midX, y: rect.minY * 0.7)
        )
        p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        p.closeSubpath()
        return p
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
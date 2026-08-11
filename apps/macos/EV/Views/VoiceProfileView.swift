import SwiftUI

struct VoiceSamplesSheet: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var playingSampleId: String?
    @State private var showResetConfirm = false

    private var coreSamples: [VoiceSample] {
        model.voiceSamples.filter { $0.tier == "core" }
    }

    private var cacheSamples: [VoiceSample] {
        model.voiceSamples.filter { $0.tier == "cache" }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if model.voiceSamples.isEmpty {
                emptyState
            } else {
                sampleList
            }
            Divider()
            footer
        }
        .frame(width: 580, height: 520)
        .onAppear { model.loadVoiceSamples() }
        .alert("重置声纹", isPresented: $showResetConfirm) {
            Button("取消", role: .cancel) {}
            Button("重置", role: .destructive) {
                model.resetVoiceProfile()
            }
        } message: {
            Text("将删除所有已收集的 \(model.voiceSamples.count) 个声纹样本，声纹模型将回到初始状态。此操作不可撤销。")
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("声纹样本")
                    .font(.headline)
                HStack(spacing: 12) {
                    Label("核心 \(coreSamples.count)", systemImage: "star.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                    Label("缓存 \(cacheSamples.count)", systemImage: "tray")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if model.voiceProfile.centroidCount > 0 {
                        Label("质心 \(model.voiceProfile.centroidCount)", systemImage: "circle.grid.3x3.fill")
                            .font(.caption)
                            .foregroundStyle(.blue)
                    }
                }
            }
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer()
            Image(systemName: "waveform")
                .font(.system(size: 40))
                .foregroundStyle(.tertiary)
            Text("暂无样本")
                .font(.headline)
            Text("正常使用 EV 时会自动收集高质量语音样本")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    private var sampleList: some View {
        ScrollView {
            LazyVStack(spacing: 0, pinnedViews: [.sectionHeaders]) {
                if !coreSamples.isEmpty {
                    Section {
                        ForEach(Array(coreSamples.enumerated()), id: \.element.id) { index, sample in
                            sampleRow(sample, index: index, tier: "core")
                            if index < coreSamples.count - 1 || !cacheSamples.isEmpty {
                                Divider().padding(.leading, 44)
                            }
                        }
                    } header: {
                        tierHeader(title: "核心样本", subtitle: "用于建模，最多20个，优先保留高质量样本", systemImage: "star.fill", color: .orange, count: coreSamples.count)
                    }
                }

                if !cacheSamples.isEmpty {
                    Section {
                        ForEach(Array(cacheSamples.enumerated()), id: \.element.id) { index, sample in
                            sampleRow(sample, index: index, tier: "cache")
                            if index < cacheSamples.count - 1 {
                                Divider().padding(.leading, 44)
                            }
                        }
                    } header: {
                        tierHeader(title: "缓存样本", subtitle: "仅记录不用于建模，最多50条", systemImage: "tray", color: .secondary, count: cacheSamples.count)
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
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor))
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
                }
                Text(formatDate(sample.createdAt))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            waveformBar(score: sample.score)

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
                togglePlay(sample)
            } label: {
                Image(systemName: isPlaying ? "stop.fill" : "play.fill")
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.borderless)

            Button(role: .destructive) {
                model.deleteVoiceSample(sample.id)
            } label: {
                Image(systemName: "trash")
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.borderless)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(isPlaying ? Color.accentColor.opacity(0.06) : Color.clear)
        .contentShape(Rectangle())
    }

    private func waveformBar(score: Double) -> some View {
        let bars = 8
        let normalizedScore = min(max(score, 0), 1)
        return HStack(spacing: 2) {
            ForEach(0..<bars, id: \.self) { i in
                let threshold = Double(i) / Double(bars - 1)
                let height = CGFloat(0.3 + 0.7 * normalizedScore * (threshold <= normalizedScore ? 1 : 0.3))
                RoundedRectangle(cornerRadius: 1)
                    .fill(threshold <= normalizedScore ? scoreColor(score: score) : Color.secondary.opacity(0.2))
                    .frame(width: 3, height: 16 * height)
            }
        }
    }

    private var footer: some View {
        HStack {
            Button(role: .destructive) {
                showResetConfirm = true
            } label: {
                Text("重置声纹")
            }
            .tint(.red)
            .disabled(model.voiceSamples.isEmpty)
            Spacer()
            Button("完成") { dismiss() }
                .buttonStyle(.borderedProminent)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private func scoreColor(score: Double) -> Color {
        if score >= 0.8 { return .green }
        if score >= 0.6 { return .orange }
        return .secondary
    }

    private func formatDuration(_ ms: Int) -> String {
        let seconds = Double(ms) / 1000.0
        if seconds < 60 {
            return String(format: "%.1fs", seconds)
        }
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

    private func togglePlay(_ sample: VoiceSample) {
        if playingSampleId == sample.id {
            model.audioPlayer.stop()
            playingSampleId = nil
        } else {
            model.audioPlayer.stop()
            playingSampleId = sample.id
            model.audioPlayer.play(url: URL(fileURLWithPath: sample.audioPath)) {
                Task { @MainActor in
                    if playingSampleId == sample.id {
                        playingSampleId = nil
                    }
                }
            }
        }
    }
}

#Preview {
    VoiceSamplesSheet()
        .environmentObject(AppModel(engine: MockEngineTransport()))
}

private class MockEngineTransport: EngineTransport {
    var onEvent: (@Sendable (EngineEnvelope) -> Void)?
    var onTermination: (@Sendable (Int32) -> Void)?
    var onStderr: (@Sendable (String) -> Void)?
    var isRunning: Bool { false }
    func start() throws {}
    func send(_ command: String, payload: [String: Any]) {}
    func shutdown() {}
}

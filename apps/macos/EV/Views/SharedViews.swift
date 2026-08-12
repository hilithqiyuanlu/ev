import SwiftUI

/// 一级按钮规范：icon + 文本、等高胶囊背景（圆角 12、浅灰底）。
/// 参照历史页「清空」/词典页「添加词语」，用于各页右上角的操作按钮。
struct CapsuleButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    var isDestructive = false

    func makeBody(configuration: Configuration) -> some View {
        let base: Color = isDestructive ? .red : .secondary
        let tint = configuration.isPressed
            ? base.opacity(0.6)
            : (isEnabled ? base : base.opacity(0.4))
        return configuration.label
            .foregroundStyle(tint)
            .contentShape(Rectangle())
            .padding(4)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.secondary.opacity(configuration.isPressed ? 0.10 : 0.06))
            )
    }
}

struct EmptyStateView: View {
    let text: String
    let systemImage: String

    init(_ text: String, systemImage: String) {
        self.text = text
        self.systemImage = systemImage
    }

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: systemImage)
                .font(.system(size: 16))
                .foregroundStyle(.tertiary)
            Text(text)
                .font(.subheadline)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
    }
}

struct StateLabel: View {
    let state: EngineState

    var body: some View {
        Label(state.title, systemImage: state.symbol)
            .foregroundStyle(state == .error ? .red : state == .speech ? .green : .primary)
    }
}

struct WaveformView: View {
    let audioLevel: Double
    let state: EngineState

    @State private var barOffsets: [Double] = [0, 0, 0, 0, 0]
    private let barCount = 5
    private let baseHeight: CGFloat = 4
    private let maxHeight: CGFloat = 36
    private let barWidth: CGFloat = 5
    private let barSpacing: CGFloat = 4

    var body: some View {
        HStack(alignment: .center, spacing: barSpacing) {
            ForEach(0..<barCount, id: \.self) { index in
                let targetHeight = maxHeight * max(0.05, audioLevel + barOffsets[index])
                let currentHeight = baseHeight + (targetHeight - baseHeight) * smoothStep(index: index)
                RoundedRectangle(cornerRadius: 2)
                    .fill(barColor)
                    .frame(width: barWidth, height: currentHeight)
                    .animation(.easeOut(duration: 0.12), value: audioLevel)
            }
        }
        .frame(width: CGFloat(barCount) * (barWidth + barSpacing) - barSpacing, height: maxHeight, alignment: .center)
        .onAppear {
            startAnimation()
        }
        .onChange(of: audioLevel) { _, _ in
            updateOffsets()
        }
    }

    private var barColor: Color {
        switch state {
        case .speech: return .green
        case .error: return .red
        case .listening, .stopping: return .accentColor
        case .loading: return .secondary
        default: return .secondary.opacity(0.4)
        }
    }

    private func smoothStep(index: Int) -> Double {
        let offset = barOffsets[index]
        return max(0.05, audioLevel + offset)
    }

    private func updateOffsets() {
        for i in 0..<barCount {
            barOffsets[i] = Double.random(in: -0.3...0.3)
        }
    }

    private func startAnimation() {
        for i in 0..<barCount {
            barOffsets[i] = Double.random(in: -0.2...0.2)
        }
    }
}

struct ActivityStatusView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 10) {
            WaveformView(audioLevel: model.audioLevel, state: model.engineState)
                .frame(height: 42)
            // 远场调试读数: 原始 dBFS + AGC 增益; 非监听态显示占位
            HStack(spacing: 10) {
                if model.isListening {
                    Text(String(format: "%.0f dB", model.rawRmsDb))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(rawDbColor)
                    Text(String(format: "×%.1f", model.agcGain))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                } else {
                    Text("—")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
            }
            Text(model.activityTitle).font(.headline)
            Text(model.activityDetail)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(model.displayTranscript)
                .font(.title3)
                .foregroundStyle(
                    model.partialText.isEmpty && model.lastFinalText.isEmpty ? .secondary : .primary
                )
                .multilineTextAlignment(.center)
                .lineLimit(4)
                .textSelection(.enabled)
                .frame(maxWidth: 640)
        }
        .padding(.horizontal, 24)
        .frame(maxWidth: .infinity, minHeight: 190, alignment: .center)
    }

    /// 原始音量颜色: >= -48dB 白色(正常); -60~-48 橙色(太轻, 可能录不上); < -60 灰色(基本没声)
    private var rawDbColor: Color {
        if model.rawRmsDb >= -48 { return .primary }
        if model.rawRmsDb >= -60 { return .orange }
        return .secondary
    }
}

struct CenteredPage<Content: View>: View {
    let maxWidth: CGFloat
    @ViewBuilder let content: Content

    init(maxWidth: CGFloat = 760, @ViewBuilder content: () -> Content) {
        self.maxWidth = maxWidth
        self.content = content()
    }

    var body: some View {
        ScrollView {
            content
                .frame(maxWidth: maxWidth)
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .center)
        }
    }
}

struct SegmentRow: View {
    @EnvironmentObject private var model: AppModel
    let segment: Segment
    var showActions = true
    var onDelete: (() -> Void)?

    private var audioFileExists: Bool {
        FileManager.default.fileExists(atPath: segment.audioPath)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Button {
                model.audioPlayer.toggle(path: segment.audioPath)
            } label: {
                Image(systemName: model.audioPlayer.playingPath == segment.audioPath ? "stop.fill" : "play.fill")
                    .frame(width: 18, height: 18)
            }
            .buttonStyle(.borderless)
            .help("回放语音")
            .disabled(!audioFileExists)

            VStack(alignment: .leading, spacing: 5) {
                Group {
                    if segment.utterances.isEmpty {
                        Text(cleanPrefix(from: segment.transcript).isEmpty ? "（无转写）" : cleanPrefix(from: segment.transcript))
                            .lineLimit(3)
                    } else {
                        VStack(alignment: .leading, spacing: 2) {
                            ForEach(segment.utterances) { utt in
                                Text(cleanPrefix(from: utt.text))
                            }
                        }
                        .lineLimit(6)
                    }
                }
                .textSelection(.enabled)
                HStack(spacing: 10) {
                    Text(timeFromISO(segment.startedAt))
                    Text(segment.speakerDisplayLabel)
                    Text(String(format: "%.1f 秒", Double(segment.durationMS) / 1000))
                    if let score = segment.speakerScore {
                        Text(String(format: "声纹 %.3f", score))
                    }
                    // 音频质量标签
                    if let snr = segment.snrDb {
                        Text(String(format: "SNR %.1f dB", snr))
                            .foregroundStyle(snrColor(snr))
                    }
                    if segment.qualityLabel != "ok" {
                        Label(
                            qualityLabelText(segment.qualityLabel),
                            systemImage: "exclamationmark.triangle.fill"
                        )
                        .foregroundStyle(.orange)
                    }
                    if segment.queryCandidate {
                        Label(segment.queryText.isEmpty ? "Query" : segment.queryText, systemImage: "bolt.fill")
                            .foregroundStyle(.green)
                    }
                    if !audioFileExists {
                        Label("原文件已删除", systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            if showActions {
                HStack(spacing: 10) {
                    Button {
                        model.openInFinder(segment.audioPath)
                    } label: {
                        Image(systemName: "folder")
                            .foregroundStyle(audioFileExists ? .primary : Color.secondary.opacity(0.5))
                    }
                    .buttonStyle(.borderless)
                    .help(audioFileExists ? "在 Finder 中显示" : "原文件已删除")
                    .disabled(!audioFileExists)

                    if onDelete != nil {
                        Button {
                            onDelete?()
                        } label: {
                            Image(systemName: "xmark")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(.primary)
                        }
                        .buttonStyle(.borderless)
                        .help("删除此记录")
                    }
                }
            }
        }
        .padding(.vertical, 5)
    }
}

struct HistoryRow: View {
    @EnvironmentObject private var model: AppModel
    let item: HistoryItem
    var onDelete: (() -> Void)?
    var onEdit: ((Segment) -> Void)?

    var body: some View {
        switch item {
        case .segment(let segment):
            segmentRow(segment)
        case .query(let query):
            queryRow(query)
        }
    }

    private func segmentRow(_ segment: Segment) -> some View {
        let audioFileExists = FileManager.default.fileExists(atPath: segment.audioPath)
        return HStack(alignment: .top, spacing: 12) {
            Button {
                model.audioPlayer.toggle(path: segment.audioPath)
            } label: {
                Image(systemName: model.audioPlayer.playingPath == segment.audioPath ? "stop.fill" : "play.fill")
                    .frame(width: 18, height: 18)
            }
            .buttonStyle(.borderless)
            .help("回放语音")
            .disabled(!audioFileExists)

            VStack(alignment: .leading, spacing: 5) {
                Group {
                    if segment.utterances.isEmpty {
                        Text(cleanPrefix(from: segment.transcript).isEmpty ? "（无转写）" : cleanPrefix(from: segment.transcript))
                            .lineLimit(3)
                    } else {
                        VStack(alignment: .leading, spacing: 2) {
                            ForEach(segment.utterances) { utt in
                                Text(cleanPrefix(from: utt.text))
                            }
                        }
                        .lineLimit(6)
                    }
                }
                .textSelection(.enabled)
                HStack(spacing: 10) {
                    Text(timeFromISO(segment.startedAt))
                    Text(segment.speakerDisplayLabel)
                    Text(String(format: "%.1f 秒", Double(segment.durationMS) / 1000))
                    if let score = segment.speakerScore {
                        Text(String(format: "声纹 %.3f", score))
                    }
                    // 音频质量标签
                    if let snr = segment.snrDb {
                        Text(String(format: "SNR %.1f dB", snr))
                            .foregroundStyle(snrColor(snr))
                    }
                    if segment.qualityLabel != "ok" {
                        Label(
                            qualityLabelText(segment.qualityLabel),
                            systemImage: "exclamationmark.triangle.fill"
                        )
                        .foregroundStyle(.orange)
                    }
                    if segment.wasCorrected {
                        Label("已修正", systemImage: "pencil.and.outline")
                            .foregroundStyle(.orange)
                    }
                    if segment.queryCandidate {
                        Label(segment.queryText.isEmpty ? "Query" : segment.queryText, systemImage: "bolt.fill")
                            .foregroundStyle(.green)
                    }
                    if !audioFileExists {
                        Label("原文件已删除", systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            HStack(spacing: 10) {
                Button {
                    onEdit?(segment)
                } label: {
                    Image(systemName: "pencil")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.primary)
                }
                .buttonStyle(.borderless)
                .help("修正转写文本")

                Button {
                    model.openInFinder(segment.audioPath)
                } label: {
                    Image(systemName: "folder")
                        .foregroundStyle(audioFileExists ? .primary : Color.secondary.opacity(0.5))
                }
                .buttonStyle(.borderless)
                .help(audioFileExists ? "在 Finder 中显示" : "原文件已删除")
                .disabled(!audioFileExists)

                Button {
                    onDelete?()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.primary)
                }
                .buttonStyle(.borderless)
                .help("删除此记录")
            }
        }
        .padding(.vertical, 5)
    }

    private func queryRow(_ query: QueryItem) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: query.source == "manual" ? "keyboard" : "mic")
                .foregroundStyle(.secondary)
                .frame(width: 18)

            VStack(alignment: .leading, spacing: 5) {
                Text(query.text)
                    .lineLimit(3)
                    .textSelection(.enabled)
                HStack(spacing: 8) {
                    Text(query.source == "manual" ? "手动" : "语音")
                        .font(.caption)
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .background(Color.secondary.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 3))
                    Text(timeFromISO(query.createdAt))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 8)
            Button {
                onDelete?()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.primary)
            }
            .buttonStyle(.borderless)
            .help("删除此记录")
        }
        .padding(.vertical, 5)
    }
}

struct QueryRow: View {
    @EnvironmentObject private var model: AppModel
    let query: QueryItem

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: query.source == "manual" ? "keyboard" : "mic")
                .foregroundStyle(.secondary)
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 4) {
                Text(query.text)
                    .textSelection(.enabled)
                HStack(spacing: 8) {
                    Text(query.source == "manual" ? "手动" : "语音")
                        .font(.caption)
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .background(Color.secondary.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 3))
                    Text(timeFromISO(query.createdAt))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            Button {
                model.deleteQuery(query.id)
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.primary)
            }
            .buttonStyle(.borderless)
            .help("删除")
        }
        .padding(.vertical, 4)
    }
}

struct ChecklistRow: View {
    let label: String
    let done: Bool
    let detail: String?

    init(_ label: String, done: Bool, detail: String? = nil) {
        self.label = label
        self.done = done
        self.detail = detail
    }

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: done ? "checkmark.circle.fill" : "circle")
                .font(.system(size: 16))
                .foregroundStyle(done ? .green : .secondary)
                .frame(width: 18, alignment: .center)
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.subheadline)
                if let detail {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
    }
}

private func qualityLabelText(_ label: String) -> String {
    switch label {
    case "rejected_low_snr": return "信噪比低"
    case "rejected_low_level": return "音量过低"
    case "rejected_non_voice": return "非人声"
    default: return label
    }
}

private func snrColor(_ snr: Double) -> Color {
    if snr >= 10 { return .green }
    if snr >= 3 { return .orange }
    return .red
}

private func cleanPrefix(from text: String) -> String {
    var result = text
    let prefixes = ["（你）：", "（他人）：", "(你)：", "(他人)："]
    for prefix in prefixes {
        if result.hasPrefix(prefix) {
            result.removeFirst(prefix.count)
            break
        }
    }
    return result.trimmingCharacters(in: .whitespaces)
}

/// 从 ISO 8601 UTC 字符串中提取用户本地时区的 "HH:mm" 部分
private func timeFromISO(_ iso: String) -> String {
    let utcFormatter = ISO8601DateFormatter()
    utcFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    if let date = utcFormatter.date(from: iso) {
        let localFormatter = DateFormatter()
        localFormatter.dateFormat = "HH:mm"
        return localFormatter.string(from: date)
    }
    // 回退：尝试无毫秒的 ISO 8601
    let fallbackFormatter = ISO8601DateFormatter()
    fallbackFormatter.formatOptions = [.withInternetDateTime]
    if let date = fallbackFormatter.date(from: iso) {
        let localFormatter = DateFormatter()
        localFormatter.dateFormat = "HH:mm"
        return localFormatter.string(from: date)
    }
    // 最终回退
    guard iso.count >= 16 else { return iso }
    let start = iso.index(iso.startIndex, offsetBy: 11)
    let end = iso.index(iso.startIndex, offsetBy: 16)
    return String(iso[start..<end])
}

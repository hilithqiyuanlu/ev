import SwiftUI

struct StateLabel: View {
    let state: EngineState

    var body: some View {
        Label(state.title, systemImage: state.symbol)
            .foregroundStyle(state == .error ? .red : state == .speech ? .green : .primary)
    }
}

struct SegmentRow: View {
    @EnvironmentObject private var model: AppModel
    let segment: Segment
    var showActions = true

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

            VStack(alignment: .leading, spacing: 5) {
                Text(segment.transcript.isEmpty ? "（无转写）" : segment.transcript)
                    .lineLimit(3)
                    .textSelection(.enabled)
                HStack(spacing: 10) {
                    Text(segment.speakerLabel)
                    Text(String(format: "%.1f 秒", Double(segment.durationMS) / 1000))
                    if let score = segment.speakerScore {
                        Text(String(format: "声纹 %.3f", score))
                    }
                    if segment.queryCandidate {
                        Label("Query", systemImage: "bolt.fill").foregroundStyle(.green)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            if showActions {
                Button {
                    model.openInFinder(segment.audioPath)
                } label: {
                    Image(systemName: "folder")
                }
                .buttonStyle(.borderless)
                .help("在 Finder 中显示")
            }
        }
        .padding(.vertical, 5)
    }
}

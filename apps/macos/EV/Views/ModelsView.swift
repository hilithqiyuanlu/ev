import SwiftUI

struct ModelsView: View {
    @EnvironmentObject private var model: AppModel

    private let names = [
        "vad": "FSMN-VAD",
        "asr_streaming": "Paraformer Streaming",
        "asr_final": "Paraformer Large",
        "speaker": "ERes2NetV2",
    ]

    var body: some View {
        CenteredPage(maxWidth: 760) {
            VStack(alignment: .leading, spacing: 20) {
                HStack(spacing: 16) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(model.allModelsReady ? "模型已就绪" : "需要准备模型")
                            .font(.title2.bold())
                        Text("固定版本 models-v0.1.0 · 约 1.82 GB")
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button {
                        if model.allModelsReady { model.verifyModels() }
                        else { model.downloadModels() }
                    } label: {
                        Label(
                            model.allModelsReady ? "重新校验" : "下载模型",
                            systemImage: model.allModelsReady ? "checkmark.arrow.trianglehead.counterclockwise" : "arrow.down.circle"
                        )
                    }
                    .buttonStyle(.borderedProminent)
                }

                if model.downloadProgress > 0 && model.downloadProgress < 1 {
                    VStack(alignment: .leading, spacing: 7) {
                        ProgressView(value: model.downloadProgress)
                        HStack {
                            Text("\(model.downloadLabel) · \(Int(model.downloadProgress * 100))%")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Button("取消", role: .cancel) { model.cancelDownload() }
                        }
                    }
                }

                Divider()

                StatusRow(
                    title: "Python 运行时",
                    detail: model.runtimeLabel,
                    ready: model.runtimeReady
                )

                Divider()

                if model.models.isEmpty {
                    ContentUnavailableView("尚未检查模型", systemImage: "shippingbox")
                        .frame(maxWidth: .infinity, minHeight: 260, alignment: .center)
                } else {
                    VStack(spacing: 0) {
                        ForEach(model.models) { item in
                            StatusRow(
                                title: names[item.key] ?? item.key,
                                detail: item.errors.isEmpty ? item.path : item.errors.joined(separator: "，"),
                                ready: item.ready
                            )
                            if item.id != model.models.last?.id { Divider() }
                        }
                    }
                }
            }
        }
        .navigationTitle("模型")
    }
}

private struct StatusRow: View {
    let title: String
    let detail: String
    let ready: Bool

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: ready ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(ready ? .green : .orange)
                .frame(width: 20)
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.headline)
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .textSelection(.enabled)
            }
            Spacer()
        }
        .padding(.vertical, 10)
    }
}

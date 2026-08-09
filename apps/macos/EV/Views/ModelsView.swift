import SwiftUI

struct ModelsView: View {
    @EnvironmentObject private var model: AppModel

    private let names = [
        "vad": "FSMN-VAD",
        "asr_streaming": "Paraformer Streaming",
        "asr_final": "SenseVoiceSmall",
        "speaker": "ERes2NetV2",
    ]

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.allModelsReady ? "模型已就绪" : "需要准备模型")
                        .font(.headline)
                    Text("固定版本 models-v0.1.0，总下载约 1.76 GB")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    if model.allModelsReady { model.verifyModels() }
                    else { model.downloadModels() }
                } label: {
                    Label(model.allModelsReady ? "重新校验" : "下载模型", systemImage: "arrow.down.circle")
                }
                if model.downloadProgress > 0 && model.downloadProgress < 1 {
                    Button("取消", role: .cancel) { model.cancelDownload() }
                }
            }
            .padding(16)

            if model.downloadProgress > 0 && model.downloadProgress < 1 {
                VStack(alignment: .leading, spacing: 5) {
                    ProgressView(value: model.downloadProgress)
                    Text("\(model.downloadLabel) · \(Int(model.downloadProgress * 100))%")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 12)
            }

            Divider()

            HStack(spacing: 12) {
                Image(systemName: model.runtimeReady ? "checkmark.circle.fill" : "exclamationmark.triangle")
                    .foregroundStyle(model.runtimeReady ? .green : .orange)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Python 运行时").font(.headline)
                    Text(model.runtimeLabel).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(16)

            Divider()

            List(model.models) { item in
                HStack(spacing: 12) {
                    Image(systemName: item.ready ? "checkmark.circle.fill" : "xmark.circle")
                        .foregroundStyle(item.ready ? .green : .red)
                    VStack(alignment: .leading, spacing: 4) {
                        Text(names[item.key] ?? item.key).font(.headline)
                        Text(item.path).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                        if !item.errors.isEmpty {
                            Text(item.errors.joined(separator: "，")).font(.caption).foregroundStyle(.red)
                        }
                    }
                    Spacer()
                }
                .padding(.vertical, 5)
            }
            .listStyle(.inset)
        }
        .navigationTitle("模型")
    }
}

import SwiftUI

struct LiveView: View {
    @EnvironmentObject private var model: AppModel
    @State private var queryText = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 14) {
                StateLabel(state: model.engineState)
                    .frame(minWidth: 95, alignment: .leading)
                Picker("输入设备", selection: $model.selectedDevice) {
                    ForEach(model.devices) { device in
                        Text(device.name).tag(device.name)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 320)
                ProgressView(value: model.audioLevel)
                    .frame(width: 120)
                Spacer()
                Button {
                    model.toggleListening()
                } label: {
                    Label(model.isListening ? "停止" : "监听", systemImage: model.isListening ? "stop.fill" : "mic.fill")
                }
                .keyboardShortcut(.space, modifiers: [.command])
                .disabled(model.engineState == .loading || model.engineState == .stopping)
            }
            .padding(16)

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Text("实时转写").font(.headline)
                Text(model.partialText.isEmpty ? "等待语音…" : model.partialText)
                    .font(.title3)
                    .foregroundStyle(model.partialText.isEmpty ? .secondary : .primary)
                    .frame(maxWidth: .infinity, minHeight: 54, alignment: .topLeading)
                    .textSelection(.enabled)
            }
            .padding(16)

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("最近语音").font(.headline)
                    Spacer()
                    Button {
                        model.loadHistory()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .buttonStyle(.borderless)
                    .help("刷新")
                }
                if model.segments.isEmpty {
                    ContentUnavailableView("暂无语音段", systemImage: "waveform.slash")
                        .frame(maxHeight: .infinity)
                } else {
                    List(Array(model.segments.prefix(8))) { segment in
                        SegmentRow(segment: segment)
                    }
                    .listStyle(.inset)
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)
            .frame(maxHeight: .infinity)

            Divider()

            HStack(spacing: 10) {
                TextField("输入 query（当前只进入待处理队列）", text: $queryText)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(submit)
                Button(action: submit) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .buttonStyle(.borderless)
                .disabled(queryText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .help("提交 Query")
            }
            .padding(16)
        }
        .navigationTitle("实时")
    }

    private func submit() {
        model.submitQuery(queryText)
        queryText = ""
    }
}

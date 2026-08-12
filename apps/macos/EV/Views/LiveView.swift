import SwiftUI

struct LiveView: View {
    @EnvironmentObject private var model: AppModel
    @State private var queryText = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 14) {
                Picker("输入设备", selection: $model.selectedDevice) {
                    ForEach(model.devices) { device in
                        Text(device.name).tag(device.name)
                    }
                }
                .frame(width: 300)

                if model.microphonePermission != .authorized {
                    HStack(spacing: 6) {
                        Image(systemName: model.microphonePermission == .notDetermined ? "mic.slash" : "mic.slash.fill")
                            .foregroundStyle(model.microphonePermission == .notDetermined ? Color.secondary : Color.red)
                        Text(model.microphonePermission == .notDetermined ? "未授权" : "已拒绝")
                            .font(.caption)
                            .foregroundStyle(model.microphonePermission == .notDetermined ? Color.secondary : Color.red)
                        if model.microphonePermission == .denied || model.microphonePermission == .restricted {
                            Button("打开系统设置") { model.openMicrophoneSettings() }
                                .buttonStyle(.borderless)
                                .font(.caption)
                        }
                    }
                }

                Spacer()

                Button {
                    model.toggleListening()
                } label: {
                    Label(
                        model.isListening ? "停止监听" : "开始监听",
                        systemImage: model.isListening ? "stop.fill" : "mic.fill"
                    )
                }
                .buttonStyle(CapsuleButtonStyle())
                .keyboardShortcut(.space, modifiers: [.command])
                .disabled(
                    model.engineState == .loading || model.engineState == .stopping ||
                    (!model.isListening && !model.canStartListening)
                )
            }
            .padding(16)

            Divider()
            ActivityStatusView()
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
                    EmptyStateView("暂无语音记录", systemImage: "waveform.slash")
                } else {
                    List(Array(model.segments.prefix(8))) { segment in
                        SegmentRow(segment: segment)
                    }
                    .listStyle(.inset)
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            HStack(spacing: 10) {
                Image(systemName: "keyboard").foregroundStyle(.secondary)
                TextField("手动输入（进入待处理队列）", text: $queryText)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(submit)
                Button(action: submit) {
                    Image(systemName: "arrow.up.circle.fill").font(.title2)
                }
                .buttonStyle(.borderless)
                .disabled(queryText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .help("提交")
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

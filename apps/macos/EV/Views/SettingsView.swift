import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Form {
            Section("输入") {
                Picker("默认设备", selection: $model.selectedDevice) {
                    ForEach(model.devices) { device in Text(device.name).tag(device.name) }
                }
            }

            Section("声纹阈值") {
                LabeledContent("用户阈值") {
                    HStack {
                        Slider(value: $model.userThreshold, in: 0.5...0.95, step: 0.01)
                        Text(model.userThreshold, format: .number.precision(.fractionLength(2))).monospacedDigit()
                    }.frame(width: 320)
                }
                LabeledContent("非用户阈值") {
                    HStack {
                        Slider(value: $model.nonUserThreshold, in: 0.1...0.7, step: 0.01)
                        Text(model.nonUserThreshold, format: .number.precision(.fractionLength(2))).monospacedDigit()
                    }.frame(width: 320)
                }
                Text("中间区域标记为 unknown；新阈值在下次开始监听时生效。")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("存储") {
                LabeledContent("数据目录", value: model.applicationSupportPath)
                LabeledContent("模型目录", value: model.applicationSupportPath + "/models")
                HStack {
                    Button("打开数据目录") { model.openApplicationSupport() }
                    Button("打开日志") { model.openLogs() }
                }
            }

            Section("系统") {
                Toggle("登录时启动 EV", isOn: Binding(
                    get: { model.launchAtLogin },
                    set: { model.setLaunchAtLogin($0) }
                ))
            }

            if !model.lastEngineLog.isEmpty {
                Section("最近引擎日志") {
                    Text(model.lastEngineLog).font(.caption.monospaced()).textSelection(.enabled)
                }
            }
        }
        .formStyle(.grouped)
        .navigationTitle("设置")
        .onChange(of: model.userThreshold) { model.saveThresholds() }
        .onChange(of: model.nonUserThreshold) { model.saveThresholds() }
    }
}

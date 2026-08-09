import SwiftUI

struct VoiceProfileView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                VStack(alignment: .leading, spacing: 5) {
                    Text("用户声纹").font(.title2.bold())
                    Text("使用当前麦克风录制 8 段，每段约 4 秒。请覆盖正常、快慢语速和不同距离。")
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            Form {
                Picker("录入设备", selection: $model.selectedDevice) {
                    ForEach(model.devices) { device in Text(device.name).tag(device.name) }
                }
                LabeledContent("声纹模型", value: "ERes2NetV2")
                LabeledContent("状态", value: model.enrollmentStatus)
            }
            .formStyle(.grouped)

            ProgressView(
                value: Double(model.enrollmentCompleted),
                total: Double(max(model.enrollmentTotal, 1))
            ) {
                Text("进度 \(model.enrollmentCompleted) / \(model.enrollmentTotal)")
            }

            HStack {
                Button("开始新的录入") { model.beginEnrollment() }
                Button {
                    model.captureEnrollmentSample()
                } label: {
                    Label("录制下一段", systemImage: "record.circle")
                }
                .disabled(model.enrollmentStatus == "尚未开始" || model.enrollmentStatus == "正在录音 4 秒" || model.enrollmentStatus == "声纹录入完成")
                Button("取消", role: .cancel) { model.cancelEnrollment() }
                Spacer()
            }
            Spacer()
        }
        .padding(24)
        .navigationTitle("声纹")
    }
}

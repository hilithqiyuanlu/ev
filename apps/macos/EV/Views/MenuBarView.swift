import AppKit
import SwiftUI

struct MenuBarView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(model.activityTitle, systemImage: model.activitySymbol)
                .font(.headline)
            Text(model.activityDetail)
                .font(.caption)
                .foregroundStyle(.secondary)
            ProgressView(value: model.audioLevel)
                .progressViewStyle(.linear)
            Button {
                model.toggleListening()
            } label: {
                Label(model.isListening ? "停止监听" : "开始监听", systemImage: model.isListening ? "stop.fill" : "mic.fill")
            }
            .disabled(
                model.engineState == .loading || model.engineState == .stopping ||
                (!model.isListening && !model.canStartListening)
            )
            Divider()
            Button("打开 EV") {
                openWindow(id: "main")
                NSApp.activate(ignoringOtherApps: true)
            }
            Button("退出") {
                model.quitApplication()
            }
        }
        .padding(14)
        .frame(width: 240)
    }
}

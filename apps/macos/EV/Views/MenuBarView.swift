import AppKit
import SwiftUI

struct MenuBarView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(model.engineState.title, systemImage: model.engineState.symbol)
                .font(.headline)
            ProgressView(value: model.audioLevel)
                .progressViewStyle(.linear)
            Button {
                model.toggleListening()
            } label: {
                Label(model.isListening ? "停止监听" : "开始监听", systemImage: model.isListening ? "stop.fill" : "mic.fill")
            }
            .disabled(model.engineState == .loading || model.engineState == .stopping)
            Divider()
            Button("打开 EV") {
                openWindow(id: "main")
                NSApp.activate(ignoringOtherApps: true)
            }
            Button("退出") {
                model.shutdown()
                NSApp.terminate(nil)
            }
        }
        .padding(14)
        .frame(width: 240)
    }
}

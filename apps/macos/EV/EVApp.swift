import SwiftUI

@main
struct EVApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup("EV", id: "main") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 920, minHeight: 620)
        }
        .defaultSize(width: 1080, height: 720)
        .commands {
            CommandGroup(after: .appInfo) {
                Button(model.isListening ? "停止监听" : "开始监听") {
                    model.toggleListening()
                }
                .keyboardShortcut("l", modifiers: [.command, .shift])
            }
        }

        MenuBarExtra("EV", systemImage: model.engineState.symbol) {
            MenuBarView()
                .environmentObject(model)
        }
        .menuBarExtraStyle(.window)
    }
}

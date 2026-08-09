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
            CommandGroup(replacing: .appTermination) {
                Button("退出 EV") {
                    model.quitApplication()
                }
                .keyboardShortcut("q")
            }
            CommandGroup(after: .appInfo) {
                Button(model.isListening ? "停止监听" : "开始监听") {
                    model.toggleListening()
                }
                .keyboardShortcut("l", modifiers: [.command, .shift])
            }
        }

        MenuBarExtra("EV", systemImage: model.activitySymbol) {
            MenuBarView()
                .environmentObject(model)
        }
        .menuBarExtraStyle(.window)
    }
}

import SwiftUI

enum AppSection: String, CaseIterable, Identifiable {
    case home = "首页"
    case history = "历史"
    case settings = "设置"
    var id: String { rawValue }

    var symbol: String {
        switch self {
        case .home: "house"
        case .history: "clock.arrow.circlepath"
        case .settings: "gearshape"
        }
    }
}

struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @State private var section: AppSection?

    var body: some View {
        NavigationSplitView {
            List(AppSection.allCases, selection: $section) { item in
                Label(item.rawValue, systemImage: item.symbol).tag(item)
            }
            .navigationSplitViewColumnWidth(min: 150, ideal: 180, max: 220)
        } detail: {
            switch section ?? (model.hasCompletedOnboarding ? .home : .settings) {
            case .home: HomeView()
            case .history: HistoryView()
            case .settings: SettingsView()
            }
        }
        .alert("EV", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("好") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
        .onAppear {
            if section == nil {
                section = model.hasCompletedOnboarding ? .home : .settings
            }
        }
        .onChange(of: model.hasCompletedOnboarding) { completed in
            if completed && section == .settings {
                section = .home
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("goHome"))) { _ in
            section = .home
        }
    }
}

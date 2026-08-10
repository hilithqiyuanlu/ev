import SwiftUI

enum AppSection: String, CaseIterable, Identifiable {
    case home = "首页"
    case history = "历史"
    case lexicon = "词典"
    case models = "模型"
    case settings = "设置"
    var id: String { rawValue }

    var symbol: String {
        switch self {
        case .home: "house"
        case .history: "clock.arrow.circlepath"
        case .lexicon: "character.book.closed"
        case .models: "cube.box"
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
            case .lexicon: LexiconView()
            case .models: ModelsView()
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
        .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("goModels"))) { _ in
            section = .models
        }
        .overlay(alignment: .bottom) {
            if model.showLearnedWordsToast, let text = model.learnedWordsToast {
                Text(text)
                    .font(.subheadline)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(.orange.opacity(0.9), in: Capsule())
                    .padding(.bottom, 16)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.25), value: model.showLearnedWordsToast)
    }
}

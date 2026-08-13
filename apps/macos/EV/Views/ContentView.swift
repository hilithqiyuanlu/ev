import SwiftUI

enum AppSection: String, CaseIterable, Identifiable {
    case home = "输入"
    case history = "历史"
    case lexicon = "词典"
    case models = "模型"
    case voice = "声纹"
    case settings = "设置"
    var id: String { rawValue }

    var symbol: String {
        switch self {
        case .home: "mic"
        case .history: "clock.arrow.circlepath"
        case .lexicon: "character.book.closed"
        case .models: "cube.box"
        case .voice: "person.wave.2"
        case .settings: "gearshape"
        }
    }
}

private enum HistorySection {
    case language
    case environment
}

struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @State private var section: AppSection = .home
    @State private var historyExpanded = true
    @State private var historySection: HistorySection = .language

    var body: some View {
        NavigationSplitView {
            List {
                sidebarButton(.home)
                historyGroupButton
                if historyExpanded {
                    historyButton("语言", symbol: "waveform", target: .language)
                    historyButton("环境", symbol: "ear", target: .environment)
                }
                ForEach([AppSection.lexicon, .models, .voice, .settings]) { sidebarButton($0) }
            }
            .navigationSplitViewColumnWidth(min: 150, ideal: 180, max: 220)
            .animation(.easeInOut(duration: 0.18), value: historyExpanded)
        } detail: {
            switch section {
            case .home: HomeView()
            case .history:
                if historySection == .language { HistoryView() } else { EnvironmentHistoryView() }
            case .lexicon: LexiconView()
            case .models: ModelsView()
            case .voice: VoiceView()
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

    private func sidebarButton(_ item: AppSection) -> some View {
        let isSelected = section == item
        return Button {
            section = item
        } label: {
            HStack(spacing: 8) {
                Image(systemName: item.symbol)
                    .frame(width: 16)
                Text(item.rawValue)
            }
            .foregroundStyle(isSelected ? Color.white : Color.primary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 5)
            .padding(.horizontal, 8)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isSelected ? Color.accentColor : Color.clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var historyGroupButton: some View {
        let isSelected = section == .history
        return Button {
            historyExpanded.toggle()
            section = .history
        } label: {
            HStack(spacing: 8) {
                Image(systemName: AppSection.history.symbol)
                    .frame(width: 16)
                Text(AppSection.history.rawValue)
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 10, weight: .semibold))
                    .rotationEffect(.degrees(historyExpanded ? 90 : 0))
            }
            .foregroundStyle(isSelected ? Color.white : Color.primary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 5)
            .padding(.horizontal, 8)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isSelected ? Color.accentColor : Color.clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func historyButton(_ title: String, symbol: String, target: HistorySection) -> some View {
        let isSelected = section == .history && historySection == target
        return Button {
            section = .history
            historySection = target
        } label: {
            HStack(spacing: 8) {
                Image(systemName: symbol)
                    .frame(width: 16)
                Text(title)
            }
            .foregroundStyle(isSelected ? Color.accentColor : Color.primary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 4)
            .padding(.leading, 24)
            .padding(.trailing, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

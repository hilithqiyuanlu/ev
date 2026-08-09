import SwiftUI

enum AppSection: String, CaseIterable, Identifiable {
    case live = "实时"
    case history = "历史"
    case voice = "声纹"
    case models = "模型"
    case settings = "设置"
    var id: String { rawValue }

    var symbol: String {
        switch self {
        case .live: "waveform"
        case .history: "clock.arrow.circlepath"
        case .voice: "person.wave.2"
        case .models: "shippingbox"
        case .settings: "gearshape"
        }
    }
}

struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @State private var section: AppSection? = .live

    var body: some View {
        NavigationSplitView {
            List(AppSection.allCases, selection: $section) { item in
                Label(item.rawValue, systemImage: item.symbol).tag(item)
            }
            .navigationSplitViewColumnWidth(min: 150, ideal: 180, max: 220)
        } detail: {
            switch section ?? .live {
            case .live: LiveView()
            case .history: HistoryView()
            case .voice: VoiceProfileView()
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
    }
}

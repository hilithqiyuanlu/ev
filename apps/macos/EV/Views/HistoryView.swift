import SwiftUI

struct HistoryView: View {
    @EnvironmentObject private var model: AppModel
    @State private var filterDate: Date?
    @State private var filterTask: Task<Void, Never>?
    @State private var speakerFilter = ""
    @State private var queryOnly = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Menu {
                    Button("全部日期") {
                        filterDate = nil
                        scheduleFilter()
                    }
                    Button("今天") {
                        filterDate = Calendar.current.startOfDay(for: Date())
                        scheduleFilter()
                    }
                    Button("昨天") {
                        filterDate = Calendar.current.date(byAdding: .day, value: -1, to: Calendar.current.startOfDay(for: Date()))
                        scheduleFilter()
                    }
                    Divider()
                    Button("选择日期...") {
                        if filterDate == nil {
                            filterDate = Date()
                            scheduleFilter()
                        }
                    }
                } label: {
                    Text(dateFilterLabel)
                        .foregroundStyle(.secondary)
                }
                .menuStyle(.borderlessButton)
                .fixedSize()

                if filterDate != nil {
                    DatePicker("", selection: Binding(
                        get: { filterDate ?? Date() },
                        set: {
                            filterDate = Calendar.current.startOfDay(for: $0)
                            scheduleFilter()
                        }
                    ), displayedComponents: .date)
                    .labelsHidden()
                    .frame(maxWidth: 130)
                }

                Menu {
                    Button("全部") {
                        speakerFilter = ""
                        scheduleFilter()
                    }
                    Button("我") {
                        speakerFilter = "user"
                        scheduleFilter()
                    }
                    Button("他人") {
                        speakerFilter = "non-user"
                        scheduleFilter()
                    }
                } label: {
                    Text(speakerFilterLabel)
                        .foregroundStyle(.secondary)
                }
                .menuStyle(.borderlessButton)
                .fixedSize()

                Toggle("", isOn: $queryOnly)
                    .toggleStyle(.checkbox)
                    .labelsHidden()
                    .help("仅显示 Query 候选")
                    .onChange(of: queryOnly) { _ in scheduleFilter() }

                Spacer()

                if model.isLoadingHistory {
                    ProgressView().controlSize(.small)
                }

                Button(role: .destructive) {
                    model.deleteAllSegments()
                    model.deleteAllQueries()
                } label: {
                    Label("清空", systemImage: "trash")
                }
                .disabled(model.historyItems.isEmpty)
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)
            .padding(.bottom, 12)

            Divider()

            if model.historyItems.isEmpty && !model.isLoadingHistory {
                EmptyStateView("暂无历史记录", systemImage: "tray")
            } else {
                List(model.historyItems) { item in
                    HistoryRow(item: item) {
                        switch item {
                        case .segment(let segment):
                            model.deleteSegment(segment.id)
                        case .query(let query):
                            model.deleteQuery(query.id)
                        }
                    }
                    .contextMenu {
                        switch item {
                        case .segment(let segment):
                            Button("在 Finder 中显示") { model.openInFinder(segment.audioPath) }
                            Divider()
                            Button("删除", role: .destructive) {
                                model.deleteSegment(segment.id)
                            }
                        case .query(let query):
                            Button("删除", role: .destructive) {
                                model.deleteQuery(query.id)
                            }
                        }
                    }
                    .swipeActions {
                        Button("删除", role: .destructive) {
                            switch item {
                            case .segment(let segment):
                                model.deleteSegment(segment.id)
                            case .query(let query):
                                model.deleteQuery(query.id)
                            }
                        }
                    }
                }
                .listStyle(.inset)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .navigationTitle("历史")
        .onAppear { applyFilters() }
    }

    private func scheduleFilter() {
        filterTask?.cancel()
        filterTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 300_000_000)
            if !Task.isCancelled { applyFilters() }
        }
    }

    private var dateFilterLabel: String {
        guard let date = filterDate else { return "全部日期" }
        let calendar = Calendar.current
        if calendar.isDateInToday(date) { return "今天" }
        if calendar.isDateInYesterday(date) { return "昨天" }
        let formatter = DateFormatter()
        formatter.dateFormat = "M月d日"
        return formatter.string(from: date)
    }

    private var speakerFilterLabel: String {
        switch speakerFilter {
        case "user": return "我"
        case "non-user": return "他人"
        default: return "全部发言人"
        }
    }

    private func applyFilters() {
        if let date = filterDate {
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyy-MM-dd"
            model.dateFilter = formatter.string(from: date)
        } else {
            model.dateFilter = ""
        }
        model.speakerFilter = speakerFilter
        model.queryOnly = queryOnly
        model.loadHistory()
    }
}

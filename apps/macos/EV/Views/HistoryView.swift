import SwiftUI

private enum DateFilterOption: String, CaseIterable, Identifiable {
    case all = "全部日期"
    case specific = "具体日期"
    var id: String { rawValue }
}

private enum SpeakerFilterOption: String, CaseIterable, Identifiable {
    case all = "全部发言人"
    case user = "我"
    case nonUser = "他人"
    var id: String { rawValue }
}

struct HistoryView: View {
    @EnvironmentObject private var model: AppModel
    @State private var filterDate: Date?
    @State private var filterTask: Task<Void, Never>?
    @State private var dateOption: DateFilterOption = .all
    @State private var speakerOption: SpeakerFilterOption = .all
    @State private var queryOnly = false
    @State private var editingSegment: Segment?
    @State private var editText = ""
    @State private var showClearConfirm = false
    @State private var showDatePickerPopover = false

    private static let dateDisplayFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy/MM/dd"
        return f
    }()

    private static let queryDisplayFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()

    private var displayedDate: Date { filterDate ?? Calendar.current.startOfDay(for: Date()) }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            filterHeader

            if model.historyItems.isEmpty && !model.isLoadingHistory {
                EmptyStateView("暂无历史记录", systemImage: "tray")
                    .frame(maxWidth: .infinity, minHeight: 280, alignment: .center)
            } else {
                List(model.historyItems) { item in
                    HistoryRow(item: item, onDelete: {
                        switch item {
                        case .segment(let segment):
                            model.deleteSegment(segment.id)
                        case .query(let query):
                            model.deleteQuery(query.id)
                        }
                    }, onEdit: { segment in
                        editText = segment.transcript
                        editingSegment = segment
                    })
                    .contextMenu {
                        switch item {
                        case .segment(let segment):
                            Button {
                                editText = segment.transcript
                                editingSegment = segment
                            } label: {
                                Label("修正转写...", systemImage: "pencil")
                            }
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
                    .swipeActions(edge: .trailing) {
                        Button("删除", role: .destructive) {
                            switch item {
                            case .segment(let segment):
                                model.deleteSegment(segment.id)
                            case .query(let query):
                                model.deleteQuery(query.id)
                            }
                        }
                    }
                    .swipeActions(edge: .leading) {
                        if case .segment(let segment) = item {
                            Button {
                                editText = segment.transcript
                                editingSegment = segment
                            } label: {
                                Label("修正", systemImage: "pencil")
                            }
                            .tint(.orange)
                        }
                    }
                    .padding(.vertical, 2)
                    .listRowBackground(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.primary.opacity(0.03))
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .stroke(Color.primary.opacity(0.04), lineWidth: 0.5)
                            )
                            .padding(.horizontal, 4)
                            .padding(.vertical, 2)
                    )
                    .listRowSeparator(.hidden)
                    .listRowInsets(EdgeInsets(top: 4, leading: 8, bottom: 4, trailing: 8))
                }
                .listStyle(.inset)
                .scrollContentBackground(.hidden)
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .navigationTitle("历史")
        .onAppear { applyFilters() }
        .sheet(item: $editingSegment) { segment in
            CorrectionEditSheet(
                originalText: segment.transcript,
                editedText: $editText,
                wasCorrected: segment.wasCorrected,
                onCancel: { editingSegment = nil },
                onSave: {
                    model.correctSegment(segment.id, correctedText: editText)
                    editingSegment = nil
                }
            )
        }
        .alert("清空历史记录", isPresented: $showClearConfirm) {
            Button("取消", role: .cancel) { showClearConfirm = false }
            Button("清空", role: .destructive) {
                showClearConfirm = false
                model.deleteAllSegments()
                model.deleteAllQueries()
            }
        } message: {
            Text("确定要删除全部历史记录和待处理输入吗？此操作不可撤销。")
        }
    }

    // MARK: - Filter Header

    private var filterHeader: some View {
        HStack(alignment: .center, spacing: 12) {
            dateCapsule
            speakerCapsule
            queryToggleCapsule
            Spacer()
            if model.isLoadingHistory {
                ProgressView().controlSize(.small)
            }
            clearCapsule
        }
    }

    private var dateCapsule: some View {
        HStack(spacing: 0) {
            Button {
                dateOption = .all
                filterDate = nil
                showDatePickerPopover = false
                scheduleFilter()
            } label: {
                Text("全部日期")
                    .font(.subheadline.weight(.medium))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .foregroundStyle(dateOption == .all ? .primary : .secondary)
                    .background(
                        GeometryReader { _ in
                            if dateOption == .all {
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(Color.primary.opacity(0.06))
                            }
                        }
                    )
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Button {
                if dateOption == .all {
                    let today = Calendar.current.startOfDay(for: Date())
                    filterDate = today
                    dateOption = .specific
                    scheduleFilter()
                } else {
                    showDatePickerPopover = true
                }
            } label: {
                Text(Self.dateDisplayFormatter.string(from: displayedDate))
                    .font(.subheadline.weight(.medium))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .foregroundStyle(dateOption == .specific ? .primary : .secondary)
                    .background(
                        GeometryReader { _ in
                            if dateOption == .specific {
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(Color.primary.opacity(0.06))
                            }
                        }
                    )
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .popover(isPresented: $showDatePickerPopover) {
                DatePicker(
                    "",
                    selection: Binding(
                        get: { filterDate ?? Calendar.current.startOfDay(for: Date()) },
                        set: { newDate in
                            let day = Calendar.current.startOfDay(for: newDate)
                            filterDate = day
                            dateOption = .specific
                            showDatePickerPopover = false
                            scheduleFilter()
                        }
                    ),
                    displayedComponents: .date
                )
                .labelsHidden()
                .datePickerStyle(.graphical)
                .frame(width: 300)
                .padding(16)
            }
        }
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.06))
        )
        .fixedSize(horizontal: true, vertical: true)
    }

    private var speakerCapsule: some View {
        HStack(spacing: 0) {
            ForEach(SpeakerFilterOption.allCases) { opt in
                Button {
                    speakerOption = opt
                    scheduleFilter()
                } label: {
                    Text(opt.rawValue)
                        .font(.subheadline.weight(.medium))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .foregroundStyle(speakerOption == opt ? .primary : .secondary)
                        .background(
                            GeometryReader { _ in
                                if speakerOption == opt {
                                    RoundedRectangle(cornerRadius: 10)
                                        .fill(Color.primary.opacity(0.06))
                                }
                            }
                        )
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.06))
        )
    }

    private var queryToggleCapsule: some View {
        Button {
            queryOnly.toggle()
            scheduleFilter()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: queryOnly ? "bolt.fill" : "bolt")
                    .font(.system(size: 11, weight: .semibold))
                Text("仅 Query")
                    .font(.subheadline.weight(.medium))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .foregroundStyle(queryOnly ? .primary : .secondary)
            .background(
                GeometryReader { _ in
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.primary.opacity(queryOnly ? 0.06 : 0.0))
                }
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.06))
        )
    }

    private var clearCapsule: some View {
        Button(role: .destructive) {
            showClearConfirm = true
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .semibold))
                Text("清空")
                    .font(.subheadline.weight(.medium))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .foregroundStyle(model.historyItems.isEmpty ? .tertiary : .secondary)
            .background(
                GeometryReader { _ in
                    EmptyView()
                }
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(model.historyItems.isEmpty)
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.06))
        )
    }

    private func scheduleFilter() {
        filterTask?.cancel()
        filterTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 120_000_000)
            if !Task.isCancelled { applyFilters() }
        }
    }

    private func applyFilters() {
        if let date = filterDate {
            model.dateFilter = Self.queryDisplayFormatter.string(from: date)
        } else {
            model.dateFilter = ""
        }
        switch speakerOption {
        case .all: model.speakerFilter = ""
        case .user: model.speakerFilter = "user"
        case .nonUser: model.speakerFilter = "non-user"
        }
        model.queryOnly = queryOnly
        model.loadHistory()
    }
}

struct CorrectionEditSheet: View {
    let originalText: String
    @Binding var editedText: String
    let wasCorrected: Bool
    let onCancel: () -> Void
    let onSave: () -> Void
    @FocusState private var isEditorFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("修正转写文本")
                    .font(.headline)
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.top, 20)
            .padding(.bottom, 8)

            Text("回放语音确认后修正，帮助 EV 更好地理解你的发音习惯。")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 20)
                .padding(.bottom, 16)

            VStack(alignment: .leading, spacing: 6) {
                Text("ASR 原文")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text(originalText.isEmpty ? "（无）" : originalText)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
                    .background(Color.secondary.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 14)

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("修正后")
                        .font(.caption.weight(.semibold))
                    if wasCorrected {
                        Text("（之前已修正过）")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                }
                TextField("", text: $editedText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.subheadline)
                    .lineLimit(3...6)
                    .padding(10)
                    .background(Color.secondary.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .focused($isEditorFocused)
            }
            .padding(.horizontal, 20)

            Spacer(minLength: 20)

            HStack {
                if wasCorrected {
                    Text("修正次数越多，EV 对你的发音越了解")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                Button("取消") { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Button("保存") { onSave() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(editedText.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 16)
        }
        .frame(width: 480)
        .onAppear { isEditorFocused = true }
    }
}

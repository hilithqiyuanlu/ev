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
                EmptyStateView("暂无语音记录", systemImage: "tray")
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
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
                    .listRowInsets(EdgeInsets(top: 4, leading: 8, bottom: 4, trailing: 8))
                }
                .listStyle(.inset)
                .scrollContentBackground(.hidden)
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .navigationTitle("语言")
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

    private var invalidSegmentCount: Int {
        model.segments.filter { $0.qualityLabel != "ok" }.count
    }

    private var filterHeader: some View {
        HStack(alignment: .center, spacing: 12) {
            dateCapsule
            speakerCapsule
            queryToggleCapsule
            Spacer()
            if model.isLoadingHistory {
                ProgressView().controlSize(.small)
            }
            clearInvalidCapsule
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
                Text("仅对话")
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

    private var clearInvalidCapsule: some View {
        Button {
            model.deleteQualityRejectedSegments()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "line.3.horizontal.decrease")
                    .font(.system(size: 11, weight: .semibold))
                Text("清空无效")
                    .font(.subheadline.weight(.medium))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .foregroundStyle(invalidSegmentCount == 0 ? .tertiary : .secondary)
            .contentShape(Rectangle())
        }
        .buttonStyle(CapsuleButtonStyle())
        .disabled(invalidSegmentCount == 0)
        .help("仅删除信噪比低、音量过低、非人声等质量不佳的录音段，保留正常录音")
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
            .contentShape(Rectangle())
        }
        .buttonStyle(CapsuleButtonStyle(isDestructive: true))
        .disabled(model.historyItems.isEmpty)
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

struct EnvironmentHistoryView: View {
    @EnvironmentObject private var model: AppModel
    @State private var filterDate: Date?
    @State private var dateOption: DateFilterOption = .all
    @State private var showDatePickerPopover = false
    @State private var showClearConfirm = false

    private static let displayDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy/MM/dd"
        return formatter
    }()

    private static let queryDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()

    private var displayedDate: Date { filterDate ?? Calendar.current.startOfDay(for: Date()) }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack(spacing: 12) {
                dateCapsule
                Spacer()
                if model.isLoadingEnvironmentHistory {
                    ProgressView().controlSize(.small)
                }
                clearCapsule
            }

            if model.environmentEvents.isEmpty && !model.isLoadingEnvironmentHistory {
                EmptyStateView("暂无环境记录", systemImage: "ear")
                    .frame(maxWidth: .infinity, minHeight: 280, alignment: .center)
            } else {
                List(model.environmentEvents) { event in
                    EnvironmentEventRow(event: event)
                        .padding(.vertical, 2)
                        .listRowBackground(Color.clear)
                        .listRowSeparator(.hidden)
                        .listRowInsets(EdgeInsets(top: 4, leading: 8, bottom: 4, trailing: 8))
                }
                .listStyle(.inset)
                .scrollContentBackground(.hidden)
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .navigationTitle("环境")
        .onAppear { applyDateFilter() }
        .alert("清空环境记录", isPresented: $showClearConfirm) {
            Button("取消", role: .cancel) {}
            Button("清空", role: .destructive) { model.clearEnvironmentHistory() }
        }
    }

    private var dateCapsule: some View {
        HStack(spacing: 0) {
            Button {
                dateOption = .all
                filterDate = nil
                showDatePickerPopover = false
                applyDateFilter()
            } label: {
                Text("全部日期")
                    .font(.subheadline.weight(.medium))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .foregroundStyle(dateOption == .all ? .primary : .secondary)
                    .background {
                        if dateOption == .all {
                            RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.06))
                        }
                    }
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Button {
                if dateOption == .all {
                    filterDate = Calendar.current.startOfDay(for: Date())
                    dateOption = .specific
                    applyDateFilter()
                } else {
                    showDatePickerPopover = true
                }
            } label: {
                Text(Self.displayDateFormatter.string(from: displayedDate))
                    .font(.subheadline.weight(.medium))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .foregroundStyle(dateOption == .specific ? .primary : .secondary)
                    .background {
                        if dateOption == .specific {
                            RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.06))
                        }
                    }
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .popover(isPresented: $showDatePickerPopover) {
                DatePicker(
                    "",
                    selection: Binding(
                        get: { displayedDate },
                        set: { date in
                            filterDate = Calendar.current.startOfDay(for: date)
                            dateOption = .specific
                            showDatePickerPopover = false
                            applyDateFilter()
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
        .background(RoundedRectangle(cornerRadius: 12).fill(Color.secondary.opacity(0.06)))
        .fixedSize(horizontal: true, vertical: true)
    }

    private var clearCapsule: some View {
        Button(role: .destructive) { showClearConfirm = true } label: {
            HStack(spacing: 6) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .semibold))
                Text("清空")
                    .font(.subheadline.weight(.medium))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .foregroundStyle(model.environmentEvents.isEmpty ? .tertiary : .secondary)
            .contentShape(Rectangle())
        }
        .buttonStyle(CapsuleButtonStyle(isDestructive: true))
        .disabled(model.environmentEvents.isEmpty)
    }

    private func applyDateFilter() {
        model.environmentDateFilter = filterDate.map(Self.queryDateFormatter.string) ?? ""
        model.loadEnvironmentHistory()
    }
}

private struct EnvironmentEventRow: View {
    @EnvironmentObject private var model: AppModel
    let event: EnvironmentEvent

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        return formatter
    }()

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: symbol)
                .font(.system(size: 15, weight: .medium))
                .frame(width: 28, height: 28)
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 5) {
                Text(model.envDisplayName(event.category))
                    .font(.body)
                Text("\(Self.timeFormatter.string(from: event.startedAt)) – \(Self.timeFormatter.string(from: event.endedAt))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text(durationText)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("\(Int(event.confidence * 100))%")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 40, alignment: .trailing)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.primary.opacity(0.03)))
    }

    private var durationText: String {
        let seconds = max(0, Int(event.durationSec.rounded()))
        return seconds >= 60 ? "\(seconds / 60)分\(seconds % 60)秒" : "\(seconds)秒"
    }

    private var symbol: String {
        switch event.category {
        case "typing": return "keyboard"
        case "music": return "music.note"
        case "background_speech": return "person.2.wave.2"
        case "alert": return "bell"
        case "animal": return "waveform"
        case "impact": return "burst"
        case "appliance": return "fan"
        default: return "waveform"
        }
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

import SwiftUI

struct HistoryView: View {
    @EnvironmentObject private var model: AppModel
    @State private var pendingDelete: Segment?
    @State private var confirmDeleteAll = false
    @State private var pendingQueryDelete: QueryItem?
    @State private var confirmDeleteAllQueries = false
    @State private var mode = "segments"

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Picker("内容", selection: $mode) {
                    Text("语音段").tag("segments")
                    Text("Query 队列").tag("queries")
                }
                .pickerStyle(.segmented)
                .frame(width: 180)
                TextField("日期 YYYY-MM-DD", text: $model.dateFilter)
                    .frame(width: 150)
                    .disabled(mode != "segments")
                Picker("说话人", selection: $model.speakerFilter) {
                    Text("全部").tag("")
                    Text("用户").tag("user")
                    Text("非用户").tag("non-user")
                    Text("未知").tag("unknown")
                }
                .frame(width: 150)
                .disabled(mode != "segments")
                Toggle("仅 Query", isOn: $model.queryOnly)
                    .toggleStyle(.checkbox)
                    .disabled(mode != "segments")
                Button("筛选") { model.loadHistory() }
                Spacer()
                Button(role: .destructive) {
                    if mode == "queries" { confirmDeleteAllQueries = true }
                    else { confirmDeleteAll = true }
                } label: {
                    Label("清空", systemImage: "trash")
                }
                .disabled(mode == "segments" ? model.segments.isEmpty : model.queries.isEmpty)
            }
            .padding(16)

            Divider()

            if mode == "queries" {
                if model.queries.isEmpty {
                    ContentUnavailableView("暂无待处理 Query", systemImage: "text.bubble")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(model.queries) { query in
                        VStack(alignment: .leading, spacing: 5) {
                            Text(query.text).textSelection(.enabled)
                            HStack {
                                Text(query.source == "manual" ? "手动" : "语音")
                                Text(query.status)
                                Text(query.createdAt)
                            }
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 5)
                        .contextMenu {
                            Button("删除", role: .destructive) { pendingQueryDelete = query }
                        }
                    }
                    .listStyle(.inset)
                }
            } else if model.segments.isEmpty {
                ContentUnavailableView("没有符合条件的语音段", systemImage: "tray")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(model.segments) { segment in
                    SegmentRow(segment: segment)
                        .contextMenu {
                            Button("在 Finder 中显示") { model.openInFinder(segment.audioPath) }
                            Divider()
                            Button("删除", role: .destructive) { pendingDelete = segment }
                        }
                        .swipeActions {
                            Button("删除", role: .destructive) { pendingDelete = segment }
                        }
                }
                .listStyle(.inset)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .navigationTitle("历史")
        .confirmationDialog(
            "删除这条语音及关联 Query？",
            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } })
        ) {
            Button("删除", role: .destructive) {
                if let segment = pendingDelete { model.deleteSegment(segment.id) }
                pendingDelete = nil
            }
        }
        .confirmationDialog("清空全部语音历史？此操作不可撤销。", isPresented: $confirmDeleteAll) {
            Button("清空全部", role: .destructive) { model.deleteAllSegments() }
        }
        .confirmationDialog(
            "删除这条 Query？",
            isPresented: Binding(get: { pendingQueryDelete != nil }, set: { if !$0 { pendingQueryDelete = nil } })
        ) {
            Button("删除", role: .destructive) {
                if let query = pendingQueryDelete { model.deleteQuery(query.id) }
                pendingQueryDelete = nil
            }
        }
        .confirmationDialog("清空全部 Query？此操作不可撤销。", isPresented: $confirmDeleteAllQueries) {
            Button("清空全部", role: .destructive) { model.deleteAllQueries() }
        }
    }
}

import SwiftUI

private enum LexiconFilter: String, CaseIterable, Identifiable {
    case all = "所有"
    case auto = "自动添加"
    case manual = "手动添加"
    var id: String { rawValue }
}

struct LexiconView: View {
    @EnvironmentObject private var model: AppModel
    @State private var filter: LexiconFilter = .all
    @State private var newLexiconWord = ""
    @State private var showAddPanel = false

    private let gridColumns = [
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12),
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                headerBar
                addPanel
                if visibleWords.isEmpty {
                    emptyState
                        .frame(maxWidth: .infinity, minHeight: 280, alignment: .center)
                } else {
                    wordGrid
                }
            }
            .padding(24)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .navigationTitle("词典")
    }

    // MARK: - Header Bar (Filter left, operations right)

    private var headerBar: some View {
        HStack(alignment: .center, spacing: 12) {
            filterBar
            addCapsule
            Spacer()
            Text("\(visibleWords.count) 词")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            let autoCount = model.lexiconWords.filter { $0.source == "auto" }.count
            if autoCount > 0 {
                clearAutoCapsule(autoCount: autoCount)
            }
        }
    }

    private var filterBar: some View {
        HStack(spacing: 0) {
            ForEach(LexiconFilter.allCases) { item in
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        filter = item
                    }
                } label: {
                    HStack(spacing: 6) {
                        if item == .auto {
                            Image(systemName: "sparkle")
                                .font(.system(size: 11, weight: .semibold))
                        } else if item == .manual {
                            Image(systemName: "feather")
                                .font(.system(size: 11, weight: .semibold))
                        }
                        Text(item.rawValue)
                            .font(.subheadline.weight(.medium))
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .foregroundStyle(filter == item ? .primary : .secondary)
                    .background(
                        GeometryReader { _ in
                            if filter == item {
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
        .fixedSize(horizontal: true, vertical: false)
    }

    // MARK: - Operation capsules (matching filter capsule visual style)

    private var learnCapsule: some View {
        Button {
            model.learnCorrections()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "arrow.triangle.2.circlepath")
                    .font(.system(size: 11, weight: .semibold))
                Text("从纠错学习")
                    .font(.subheadline.weight(.medium))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .foregroundStyle(.secondary)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.06))
        )
    }

    private var addCapsule: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.2)) {
                showAddPanel.toggle()
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "plus")
                    .font(.system(size: 11, weight: .semibold))
                Text("添加词语")
                    .font(.subheadline.weight(.medium))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .foregroundStyle(.secondary)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.06))
        )
    }

    private func clearAutoCapsule(autoCount: Int) -> some View {
        Button(role: .destructive) {
            model.clearAutoLexicon()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .semibold))
                Text("清空自动词")
                    .font(.subheadline.weight(.medium))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .foregroundStyle(.secondary)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.06))
        )
    }

    // MARK: - Add Panel

    @ViewBuilder
    private var addPanel: some View {
        if showAddPanel {
            HStack(alignment: .center, spacing: 10) {
                TextField("添加词语（如：网易云、张三、vibe coding）", text: $newLexiconWord)
                    .textFieldStyle(.roundedBorder)
                    .font(.subheadline)
                    .onSubmit { addLexiconWord() }

                Button {
                    addLexiconWord()
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(.primary)
                }
                .buttonStyle(.borderless)
                .disabled(newLexiconWord.trimmingCharacters(in: .whitespaces).isEmpty)
                .help("添加")

                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showAddPanel = false
                        newLexiconWord = ""
                    }
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.borderless)
                .help("取消")
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.secondary.opacity(0.06))
            )
            .transition(.move(edge: .top).combined(with: .opacity))
        }
    }

    // MARK: - Empty

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "character.book.closed")
                .font(.system(size: 16))
                .foregroundStyle(.tertiary)
            Text(emptyDescription)
                .font(.subheadline)
                .foregroundStyle(.tertiary)
            if filter != .all {
                Button("查看所有词语") {
                    withAnimation { filter = .all }
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.blue)
                .padding(.top, 4)
            } else if !showAddPanel {
                Button("添加第一个词") {
                    withAnimation { showAddPanel = true }
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.blue)
                .padding(.top, 4)
            }
        }
    }

    private var emptyDescription: String {
        switch filter {
        case .all: return "暂无词语，添加常用词以提升识别效果"
        case .auto: return "还没有自动学习到的词语"
        case .manual: return "还没有手动添加的词语"
        }
    }

    // MARK: - Grid

    private var wordGrid: some View {
        LazyVGrid(columns: gridColumns, alignment: .leading, spacing: 12) {
            ForEach(visibleWords) { item in
                wordCard(item)
            }
        }
    }

    @ViewBuilder
    private func wordCard(_ item: LexiconItem) -> some View {
        HStack(spacing: 10) {
            sourceIcon(item)
            Text(item.word)
                .font(.body)
                .foregroundStyle(.primary)
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer(minLength: 6)
            if item.source != "system" {
                Button {
                    model.deleteLexiconWord(item.id)
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.primary)
                }
                .buttonStyle(.borderless)
                .help("删除此词")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.primary.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.primary.opacity(0.04), lineWidth: 0.5)
        )
        .contextMenu {
            if item.source == "auto" {
                Button("提升为手动词") {
                    model.updateLexiconWord(item.id, promoteToManual: true)
                }
            }
            if item.source != "system" {
                Button(role: .destructive) {
                    model.deleteLexiconWord(item.id)
                } label: {
                    Label("删除", systemImage: "trash")
                }
            }
        }
    }

    @ViewBuilder
    private func sourceIcon(_ item: LexiconItem) -> some View {
        switch item.source {
        case "manual":
            Image(systemName: "feather")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary.opacity(0.8))
        case "system":
            Image(systemName: "gearshape")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary.opacity(0.8))
        default:
            Image(systemName: "sparkle")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Color(red: 0.25, green: 0.70, blue: 0.55))
        }
    }

    // MARK: - Data (system words excluded from display)

    private var visibleWords: [LexiconItem] {
        let nonSystem = model.lexiconWords.filter { $0.source != "system" }
        switch filter {
        case .all:
            return nonSystem.sorted { a, b in
                let rankA = sourceRank(a.source)
                let rankB = sourceRank(b.source)
                if rankA != rankB { return rankA < rankB }
                if a.source == "manual" { return a.createdAt > b.createdAt }
                return a.useCount > b.useCount
            }
        case .auto:
            return nonSystem
                .filter { $0.source == "auto" }
                .sorted { $0.useCount > $1.useCount }
        case .manual:
            return nonSystem
                .filter { $0.source == "manual" }
                .sorted { $0.createdAt > $1.createdAt }
        }
    }

    private func sourceRank(_ source: String) -> Int {
        switch source {
        case "auto": return 0
        case "manual": return 1
        default: return 2
        }
    }

    private func addLexiconWord() {
        let word = newLexiconWord.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !word.isEmpty else { return }
        model.addLexiconWord(word)
        newLexiconWord = ""
    }
}

import SwiftUI

struct ModelsView: View {
    @EnvironmentObject private var model: AppModel
    @State private var modelToDelete: AvailableModel?

    private let typeOrder: [String] = [
        "vad",
        "speech_enhancement",
        "asr_final",
        "speaker",
        "environment",
    ]

    private let typeNames: [String: String] = [
        "vad": "语音活动检测（VAD）",
        "speech_enhancement": "语音增强/降噪",
        "asr_final": "自动语音识别（ASR）",
        "speaker": "声纹识别",
        "environment": "环境感知",
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                headerSection
                modelLibraryContent
            }
            .padding(24)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .navigationTitle("模型")
        .confirmationDialog(
            "卸载模型",
            isPresented: Binding(
                get: { modelToDelete != nil },
                set: { if !$0 { modelToDelete = nil } }
            ),
            titleVisibility: .visible,
            presenting: modelToDelete
        ) { item in
            Button("卸载 \(item.name)", role: .destructive) {
                model.uninstallModel(key: item.key)
                modelToDelete = nil
            }
            Button("取消", role: .cancel) {
                modelToDelete = nil
            }
        } message: { item in
            Text("将删除本地模型文件并释放其槽位绑定。此操作不可撤销。")
        }
        .onAppear {
            model.refreshModelCatalog()
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        HStack(alignment: .center, spacing: 16) {
            HStack(spacing: 8) {
                Circle()
                    .fill(model.allModelsReady ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)
                Text(model.allModelsReady ? "已就绪" : "待配置")
                    .font(.title3.bold())
                if model.isVerifyingModels || model.downloadProgress > 0 {
                    ProgressView().controlSize(.small)
                }
            }
            Spacer()
            Button {
                model.reloadRegistry()
                model.verifyModels()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 11, weight: .semibold))
                    Text("刷新校验")
                        .font(.subheadline.weight(.medium))
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .contentShape(Rectangle())
            }
            .buttonStyle(CapsuleButtonStyle())
            .disabled(model.isVerifyingModels || model.downloadProgress > 0)
        }
    }

    // MARK: - Model Library

    private var modelLibraryContent: some View {
        VStack(alignment: .leading, spacing: 16) {
            if model.availableModels.isEmpty {
                EmptyStateView("尚未加载模型目录", systemImage: "shippingbox")
                    .frame(maxWidth: .infinity, minHeight: 280, alignment: .center)
            } else {
                let installedKeys = Set(model.installedModels.map(\.key))
                let grouped = Dictionary(grouping: model.availableModels) { $0.type }
                let sortedTypes = grouped.keys.sorted { a, b in
                    let idxA = typeOrder.firstIndex(of: a) ?? Int.max
                    let idxB = typeOrder.firstIndex(of: b) ?? Int.max
                    return idxA < idxB
                }

                ForEach(sortedTypes, id: \.self) { type in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(alignment: .center, spacing: 12) {
                            Text(typeNames[type, default: type])
                                .font(.caption.weight(.medium))
                                .foregroundStyle(.secondary)
                            Spacer()
                        }
                        slotIssue(for: type)

                        ForEach(grouped[type] ?? []) { item in
                            modelRow(for: item, isInstalled: installedKeys.contains(item.key))
                        }
                    }
                }
            }
        }
    }

    /// 已分配模型但校验未通过时，在标题行下方给出具体原因。
    @ViewBuilder
    private func slotIssue(for type: String) -> some View {
        if let assignment = model.slotAssignments.first(where: { $0.slot == type }),
           assignment.modelKey != nil,
           !assignment.status.ready {
            let reasons = assignment.status.errors
            Text(reasons.isEmpty ? "校验失败，请点击「刷新校验」" : reasons.joined(separator: "，"))
                .font(.caption)
                .foregroundStyle(.orange)
                .padding(.leading, 2)
        }
    }

    private func modelRow(for item: AvailableModel, isInstalled: Bool) -> some View {
        let isDownloading = model.downloadingKey == item.key
        let progress = model.downloadProgressByKey[item.key] ?? 0
        let message = model.downloadMessageByKey[item.key] ?? "正在下载..."
        let totalSize = model.downloadSizeByKey[item.key] ?? Int64(item.estimatedSizeBytes)
        let downloadedBytes = Int64(Double(totalSize) * progress)
        let speed = model.downloadSpeedByKey[item.key] ?? 0
        let isIndeterminate = model.downloadIndeterminateByKey[item.key] ?? true
        let source = model.downloadSourceByKey[item.key] ?? item.source

        return VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Image(systemName: isInstalled ? "checkmark.circle.fill" : (isDownloading ? "arrow.down.circle" : "circle"))
                    .foregroundStyle(isInstalled ? .green : (isDownloading ? .blue : .secondary))
                    .font(.system(size: 18))
                    .frame(width: 20)

                VStack(alignment: .leading, spacing: 4) {
                    Text(item.name)
                        .font(.headline)
                    if !item.description.isEmpty {
                        Text(item.description)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                    HStack(spacing: 8) {
                        Text(source == "modelscope" ? "ModelScope" : "GitHub")
                            .font(.caption2.weight(.medium))
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Color.secondary.opacity(0.12))
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                        if isInstalled, let installed = model.installedModels.first(where: { $0.key == item.key }) {
                            Text(ByteCountFormatter.string(fromByteCount: Int64(installed.sizeBytes), countStyle: .binary))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else if item.estimatedSizeBytes > 0 {
                            Text("约 \(ByteCountFormatter.string(fromByteCount: Int64(item.estimatedSizeBytes), countStyle: .binary))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        if item.minMemoryGb > 0 {
                            Text("需 \(item.minMemoryGb, specifier: "%.0f")GB 内存")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                Spacer()

                if isDownloading {
                    Button {
                        model.cancelDownload()
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "xmark")
                                .font(.system(size: 11, weight: .semibold))
                            Text("取消")
                                .font(.subheadline.weight(.medium))
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(CapsuleButtonStyle())
                } else if isInstalled {
                    Button {
                        modelToDelete = item
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "trash")
                                .font(.system(size: 11, weight: .semibold))
                            Text("卸载")
                                .font(.subheadline.weight(.medium))
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(CapsuleButtonStyle(isDestructive: true))
                } else {
                    Button {
                        model.installModel(key: item.key)
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "arrow.down.circle")
                                .font(.system(size: 11, weight: .semibold))
                            Text("下载")
                                .font(.subheadline.weight(.medium))
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(CapsuleButtonStyle())
                }
            }

            if isDownloading {
                VStack(alignment: .leading, spacing: 6) {
                    if isIndeterminate || progress <= 0.01 {
                        ProgressView()
                            .progressViewStyle(.linear)
                    } else {
                        ProgressView(value: progress)
                            .progressViewStyle(.linear)
                    }
                    HStack {
                        Text(message)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        if !isIndeterminate && progress > 0.01 {
                            Text("\(ByteCountFormatter.string(fromByteCount: downloadedBytes, countStyle: .binary)) / \(ByteCountFormatter.string(fromByteCount: totalSize, countStyle: .binary))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            if speed > 0 {
                                Text("• \(formatSpeed(speed))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        } else {
                            Text("约 \(ByteCountFormatter.string(fromByteCount: totalSize, countStyle: .binary))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(isDownloading ? Color.blue.opacity(0.04) : Color.primary.opacity(0.03))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(isDownloading ? Color.blue.opacity(0.2) : Color.primary.opacity(0.04), lineWidth: 0.5)
                )
        )
    }

    private func formatSpeed(_ bytesPerSec: Double) -> String {
        if bytesPerSec < 1024 {
            return "\(Int(bytesPerSec)) B/s"
        } else if bytesPerSec < 1024 * 1024 {
            return String(format: "%.1f KB/s", bytesPerSec / 1024)
        } else {
            return String(format: "%.1f MB/s", bytesPerSec / 1024 / 1024)
        }
    }
}


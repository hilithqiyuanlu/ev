import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        CenteredPage(maxWidth: 720) {
            VStack(alignment: .leading, spacing: 22) {
                storageSection
                Divider()
                systemSection
            }
        }
        .navigationTitle("设置")
    }

    // MARK: - Storage

    private var storageSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("存储", systemImage: "folder.fill")
            HStack(spacing: 10) {
                Text("数据目录:").font(.subheadline)
                Text(model.applicationSupportPath)
                    .font(.caption).foregroundStyle(.secondary)
                    .textSelection(.enabled)
                Button("打开") { model.openApplicationSupport() }
            }
            HStack(spacing: 10) {
                Text("日志目录:").font(.subheadline)
                Button("打开日志") { model.openLogs() }
            }
        }
    }

    // MARK: - System

    private var systemSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("系统", systemImage: "desktopcomputer")
            HStack(spacing: 10) {
                systemToggleCapsule(
                    label: "登录时启动 EV",
                    systemImage: "power",
                    isOn: model.launchAtLogin
                ) { model.setLaunchAtLogin($0) }
                systemToggleCapsule(
                    label: "Ctrl+T 启用/停止监听",
                    systemImage: "keyboard",
                    isOn: model.ctrlTToggleListening
                ) { model.setCtrlTToggleListening($0) }
            }
        }
    }

    /// 参考历史页「仅 Query」的开关胶囊：每个独立按钮，开启时高亮填充 + 主色文字。
    private func systemToggleCapsule(
        label: String,
        systemImage: String,
        isOn: Bool,
        onToggle: @escaping (Bool) -> Void
    ) -> some View {
        Button {
            withAnimation(.easeInOut(duration: 0.15)) { onToggle(!isOn) }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: systemImage)
                    .font(.system(size: 11, weight: .semibold))
                Text(label)
                    .font(.subheadline.weight(.medium))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .foregroundStyle(isOn ? .primary : .secondary)
            .background(
                GeometryReader { _ in
                    if isOn {
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color.primary.opacity(0.06))
                    }
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

    // MARK: - Shared helpers

    private func sectionHeader(_ title: String, systemImage: String) -> some View {
        Label(title, systemImage: systemImage)
            .font(.headline)
    }
}
import SwiftUI

struct HomeView: View {
    @EnvironmentObject private var model: AppModel
    @State private var queryText = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            topBar
            ActivityStatusView()
            querySection
            inputBar
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .navigationTitle("首页")
        .onAppear { model.loadHistory() }
    }

    // MARK: - Top Bar

    private var topBar: some View {
        HStack(spacing: 14) {
            deviceMenu

            if model.microphonePermission != .authorized {
                HStack(spacing: 6) {
                    Image(systemName: model.microphonePermission == .notDetermined ? "mic.slash" : "mic.slash.fill")
                        .foregroundStyle(model.microphonePermission == .notDetermined ? Color.secondary : Color.red)
                    Text(model.microphonePermission == .notDetermined ? "未授权" : "已拒绝")
                        .font(.caption)
                        .foregroundStyle(model.microphonePermission == .notDetermined ? Color.secondary : Color.red)
                    if model.microphonePermission == .denied || model.microphonePermission == .restricted {
                        Button("打开系统设置") { model.openMicrophoneSettings() }
                            .buttonStyle(.borderless)
                            .font(.caption)
                    }
                }
            }

            Spacer()

            listeningButton
        }
    }

    private var deviceMenu: some View {
        Menu {
            if model.devicePickerItems.isEmpty {
                Text("无可用设备")
            }
            ForEach(model.devicePickerItems, id: \.tag) { item in
                Button {
                    model.selectedDevice = item.tag
                } label: {
                    HStack {
                        Text(item.label)
                        Spacer()
                        if model.selectedDevice == item.tag || (model.selectedDevice.isEmpty && item.isDefault) {
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "mic.fill")
                    .font(.system(size: 11, weight: .semibold))
                Text(model.selectedDeviceLabel)
                    .font(.subheadline.weight(.medium))
                    .lineLimit(1)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .foregroundStyle(model.devices.isEmpty ? .tertiary : .primary)
            .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .disabled(model.isListening || model.devices.isEmpty)
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.06))
                .contentShape(Rectangle())
        )
        .help(
            model.isListening
                ? "监听中无法切换设备，请先停止监听"
                : "选择麦克风输入源；默认跟随系统设置（插上 DJI Mic Mini 时会自动切换）"
        )
    }

    private var listeningButton: some View {
        Button {
            model.toggleListening()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: model.isListening ? "stop.fill" : "mic.fill")
                    .font(.system(size: 11, weight: .semibold))
                Text(model.isListening ? "停止监听" : "开始监听")
                    .font(.subheadline.weight(.medium))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(CapsuleButtonStyle())
        .keyboardShortcut(.space, modifiers: [.command])
        .disabled(
            model.engineState == .loading || model.engineState == .stopping ||
            (!model.isListening && !model.canStartListening)
        )
    }

    // MARK: - Query Section

    private var querySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("待处理输入")
                    .font(.headline)
                Spacer()
                if model.isProcessing {
                    Label("处理中", systemImage: "hourglass")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if model.queries.isEmpty {
                EmptyStateView(
                    model.isListening ? "正在监听，说 \"小E，...\" 产生输入" : "暂无待处理输入",
                    systemImage: "text.bubble"
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            } else {
                List(model.queries) { query in
                    QueryRow(query: query)
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
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        HStack(spacing: 10) {
            Image(systemName: "keyboard")
                .foregroundStyle(.secondary)
            TextField("手动输入...", text: $queryText)
                .textFieldStyle(.roundedBorder)
                .onSubmit(submit)
            Button(action: submit) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
            }
            .buttonStyle(.borderless)
            .disabled(queryText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            .help("提交")
        }
    }

    // MARK: - Actions

    private func submit() {
        model.submitQuery(queryText)
        queryText = ""
    }
}

import AVFoundation

@MainActor
final class AudioPlayer: ObservableObject {
    @Published var playingPath: String?
    private var player: AVAudioPlayer?

    func toggle(path: String) {
        if playingPath == path {
            player?.stop()
            playingPath = nil
            return
        }
        do {
            player = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: path))
            player?.play()
            playingPath = path
        } catch {
            playingPath = nil
        }
    }
}

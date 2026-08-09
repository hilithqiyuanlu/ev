import AVFoundation
import Foundation

@MainActor
final class AudioPlayer: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published var playingPath: String?
    private var player: AVAudioPlayer?
    private var completion: (() -> Void)?

    func toggle(path: String) {
        if playingPath == path {
            stop()
            return
        }
        playFile(path)
    }

    private func playFile(_ path: String) {
        stop()
        do {
            player = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: path))
            player?.delegate = self
            player?.play()
            playingPath = path
        } catch {
            playingPath = nil
        }
    }

    func play(url: URL, completion: @escaping () -> Void) {
        stop()
        self.completion = completion
        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.delegate = self
            player?.play()
            playingPath = url.path
        } catch {
            playingPath = nil
            self.completion = nil
            completion()
        }
    }

    func stop() {
        player?.stop()
        player = nil
        playingPath = nil
        let cb = completion
        completion = nil
        cb?()
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.player = nil
            self.playingPath = nil
            let cb = self.completion
            self.completion = nil
            cb?()
        }
    }
}

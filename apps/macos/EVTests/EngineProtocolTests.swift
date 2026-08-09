import Foundation
import Testing
@testable import EV

@Test func decodesEngineEnvelope() throws {
    let data = Data(#"{"version":1,"request_id":"r","type":"engine_state","timestamp":"now","payload":{"state":"listening"}}"#.utf8)
    let envelope = try JSONDecoder().decode(EngineEnvelope.self, from: data)
    #expect(envelope.version == 1)
    #expect(envelope.requestID == "r")
    #expect(envelope.payload["state"]?.string == "listening")
}

@Test func decodesSegmentFlagsFromSQLiteNumbers() {
    let segment = Segment([
        "id": .string("segment"),
        "started_at": .string("2026-08-09T00:00:00Z"),
        "audio_path": .string("/tmp/a.wav"),
        "wake_detected": .number(1),
        "query_candidate": .number(1),
    ])
    #expect(segment?.wakeDetected == true)
    #expect(segment?.queryCandidate == true)
}

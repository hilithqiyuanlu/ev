import AppKit
import Foundation

let output = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

for size in [16, 32, 64, 128, 256, 512, 1024] {
    let image = NSImage(size: NSSize(width: size, height: size))
    image.lockFocus()
    let scale = CGFloat(size) / 1024
    let canvas = NSRect(x: 0, y: 0, width: size, height: size)
    NSColor(calibratedRed: 0.08, green: 0.09, blue: 0.10, alpha: 1).setFill()
    canvas.fill()

    NSColor(calibratedRed: 0.20, green: 0.82, blue: 0.55, alpha: 1).setFill()
    let barWidths: [CGFloat] = [38, 38, 38, 38]
    let heights: [CGFloat] = [170, 300, 430, 250]
    for index in 0..<4 {
        let rect = NSRect(
            x: (128 + CGFloat(index) * 68) * scale,
            y: (512 - heights[index] / 2) * scale,
            width: barWidths[index] * scale,
            height: heights[index] * scale
        )
        NSBezierPath(roundedRect: rect, xRadius: 19 * scale, yRadius: 19 * scale).fill()
    }

    let paragraph = NSMutableParagraphStyle()
    paragraph.alignment = .center
    let attributes: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: 310 * scale, weight: .bold),
        .foregroundColor: NSColor.white,
        .paragraphStyle: paragraph,
    ]
    NSString(string: "EV").draw(
        in: NSRect(x: 350 * scale, y: 335 * scale, width: 570 * scale, height: 390 * scale),
        withAttributes: attributes
    )
    image.unlockFocus()

    guard let tiff = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiff),
          let png = bitmap.representation(using: .png, properties: [:]) else {
        fatalError("无法生成 \(size)px 图标")
    }
    try png.write(to: output.appendingPathComponent("icon-\(size).png"))
}

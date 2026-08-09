// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "EV",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "EV", targets: ["EV"])],
    targets: [
        .executableTarget(
            name: "EV",
            path: "EV",
            exclude: ["Info.plist", "Assets.xcassets"]
        ),
        .testTarget(name: "EVTests", dependencies: ["EV"], path: "EVTests"),
    ]
)

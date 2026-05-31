import Foundation
import AVFoundation
import UIKit

@MainActor
final class CameraCaptureController: NSObject {
    static let shared = CameraCaptureController()

    private var session: AVCaptureSession?
    private var photoOutput: AVCapturePhotoOutput?
    private var continuation: CheckedContinuation<String, Never>?

    func capture() async -> String {
        #if targetEnvironment(simulator)
        return "Camera unavailable in simulator."
        #else
        return await withCheckedContinuation { (cont: CheckedContinuation<String, Never>) in
            self.continuation = cont
            Task { [weak self] in
                await self?.performCapture()
            }
        }
        #endif
    }

    private func performCapture() async {
        let session = AVCaptureSession()
        session.sessionPreset = .photo

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
                ?? AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device) else {
            finish("No camera input available.")
            return
        }

        session.beginConfiguration()
        if session.canAddInput(input) { session.addInput(input) }
        let output = AVCapturePhotoOutput()
        if session.canAddOutput(output) { session.addOutput(output) }
        session.commitConfiguration()

        self.session = session
        self.photoOutput = output

        session.startRunning()
        try? await Task.sleep(for: .milliseconds(600))

        let settings = AVCapturePhotoSettings()
        output.capturePhoto(with: settings, delegate: PhotoDelegate(owner: self))
    }

    fileprivate func didCapture(data: Data?, error: Error?) {
        session?.stopRunning()
        session = nil
        photoOutput = nil

        if let error {
            finish("Camera error: \(error.localizedDescription)")
            return
        }
        guard let data, let image = UIImage(data: data) else {
            finish("Couldn't read captured image.")
            return
        }
        let size = image.size
        let bytes = data.count
        let kb = Double(bytes) / 1024.0
        finish(String(format: "Captured image (%.0f×%.0f, %.0f KB).", size.width, size.height, kb))
    }

    private func finish(_ result: String) {
        MainActor.preconditionIsolated()
        continuation?.resume(returning: result)
        continuation = nil
    }
}

@MainActor
final class PhotoDelegate: NSObject, AVCapturePhotoCaptureDelegate {
    /// Concurrency contract: delegate callbacks may arrive on arbitrary queues; this type
    /// immediately hops to MainActor before mutating capture controller state.
    private let owner: CameraCaptureController
    init(owner: CameraCaptureController) { self.owner = owner }

    nonisolated func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let data = photo.fileDataRepresentation()
        Task { @MainActor [owner] in
            MainActor.preconditionIsolated()
            owner.didCapture(data: data, error: error)
        }
    }
}

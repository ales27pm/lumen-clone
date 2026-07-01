import UIKit
#if canImport(MSAL)
import MSAL
#endif

class LumenAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        MetricKitDiagnosticsSubscriber.shared.register()
        return true
    }

    func application(
        _ app: UIApplication,
        open url: URL,
        options: [UIApplication.OpenURLOptionsKey: Any] = [:]
    ) -> Bool {
        #if canImport(MSAL)
        return MSALPublicClientApplication.handleMSALResponse(
            url,
            sourceApplication: options[UIApplication.OpenURLOptionsKey.sourceApplication] as? String
        )
        #else
        return false
        #endif
    }

    func applicationWillResignActive(_ application: UIApplication) {
        Task { @MainActor in SceneTransitionCoordinator.shared.handleWillResignActive() }
    }

    func applicationDidBecomeActive(_ application: UIApplication) {
        Task { @MainActor in SceneTransitionCoordinator.shared.requestForegroundActivation() }
    }

    func applicationDidEnterBackground(_ application: UIApplication) {
        Task { @MainActor in SceneTransitionCoordinator.shared.handleDidEnterBackground() }
    }

    func applicationWillTerminate(_ application: UIApplication) {
        AppCancellationBus.shared.markProcessExitRequested("will-terminate")
    }

    func applicationDidReceiveMemoryWarning(_ application: UIApplication) {
        Task { @MainActor in
            await MemoryPressureMonitor.shared.handleWarning()
        }
    }
}

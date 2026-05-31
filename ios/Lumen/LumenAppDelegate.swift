import UIKit
#if canImport(MSAL)
import MSAL
#endif

class LumenAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        true
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

    func applicationDidReceiveMemoryWarning(_ application: UIApplication) {
        Task { @MainActor in
            FleetRuntimeCleanup.unloadOptionalChatSlots()
        }
    }
}

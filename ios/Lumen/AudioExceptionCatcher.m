#import "AudioExceptionCatcher.h"

@implementation AudioExceptionCatcher

+ (BOOL)tryBlock:(LumenAudioExceptionWork)work error:(NSString * _Nullable * _Nullable)error {
    // This guard only catches Objective-C NSException failures from AVAudio APIs.
    // Mach faults, Swift fatalError, and force-unwrap traps remain outside this boundary.
    @try {
        work();
        return YES;
    } @catch (NSException *exception) {
        if (error != nil) {
            NSString *reason = exception.reason ?: @"unknown";
            *error = [NSString stringWithFormat:@"%@: %@", exception.name, reason];
        }
        return NO;
    }
}

@end

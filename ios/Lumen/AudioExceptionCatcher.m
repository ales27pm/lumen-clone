#import "AudioExceptionCatcher.h"

@implementation AudioExceptionCatcher

+ (BOOL)tryBlock:(LumenAudioExceptionWork)work error:(NSString * _Nullable * _Nullable)error {
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

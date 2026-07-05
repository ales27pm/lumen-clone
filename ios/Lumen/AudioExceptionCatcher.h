#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef void (^LumenAudioExceptionWork)(void);

@interface AudioExceptionCatcher : NSObject

+ (BOOL)tryBlock:(LumenAudioExceptionWork)work error:(NSString * _Nullable * _Nullable)error;

@end

NS_ASSUME_NONNULL_END

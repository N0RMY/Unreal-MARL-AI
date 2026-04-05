#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "SocketClient.generated.h"

class FSocket;

UCLASS(BlueprintType)
class KURSOVA2_API USocketClient : public UObject
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "Socket")
    bool ConnectToServer(const FString& Host, int32 Port);

    UFUNCTION(BlueprintCallable, Category = "Socket")
    bool SendJsonAndReceiveJson(const FString& OutJson, FString& InJson);

    UFUNCTION(BlueprintCallable, Category = "RL")
    FString BuildStepJson(const FVector& HiderLoc, const FVector& SeekerLoc, const FVector& HiderVel, const FVector& SeekerVel, const TArray<float>& HiderSensors, const TArray<float>& SeekerSensors, float Reward, bool bDone, float HiderYaw, float SeekerYaw);

    // Повертає 4 дробових значення float
    UFUNCTION(BlueprintCallable, Category = "RL")
    bool ExtractAction(const FString& InJson, float& OutHiderMove, float& OutHiderTurn, float& OutSeekerMove, float& OutSeekerTurn);

    // Один виклик для 4 вихідних float
    UFUNCTION(BlueprintCallable, Category = "RL")
    bool Step(const FVector& HiderLoc, const FVector& SeekerLoc, const FVector& HiderVel, const FVector& SeekerVel, const TArray<float>& HiderSensors, const TArray<float>& SeekerSensors, float Reward, bool bDone, float HiderYaw, float SeekerYaw, float& OutHiderMove, float& OutHiderTurn, float& OutSeekerMove, float& OutSeekerTurn, FString& OutResponseJson);

    UFUNCTION(BlueprintCallable, Category = "Socket")
    void Close();

private:
    FSocket* Socket = nullptr;
};

#include "SocketClient.h"

#include "Sockets.h"
#include "SocketSubsystem.h"
#include "Common/TcpSocketBuilder.h"
#include "Interfaces/IPv4/IPv4Address.h"

#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonWriter.h"
#include "Serialization/JsonSerializer.h"

#include "Windows/AllowWindowsPlatformTypes.h"
#include <winsock2.h>
#include "Windows/HideWindowsPlatformTypes.h"

static constexpr int32  kMaxPacketBytes = 10 * 1024 * 1024;
static constexpr double kWaitSeconds = 2.0;

static bool WaitReadable(FSocket* S, double Seconds)
{
    return S && S->Wait(ESocketWaitConditions::WaitForRead, FTimespan::FromSeconds(Seconds));
}
static bool WaitWritable(FSocket* S, double Seconds)
{
    return S && S->Wait(ESocketWaitConditions::WaitForWrite, FTimespan::FromSeconds(Seconds));
}

static bool SendAll(FSocket* S, const uint8* Data, int32 Len)
{
    if (!S || !Data || Len < 0) return false;
    if (Len == 0) return true;

    int32 SentTotal = 0;
    while (SentTotal < Len)
    {
        if (!WaitWritable(S, kWaitSeconds)) return false;

        int32 JustSent = 0;
        if (!S->Send(Data + SentTotal, Len - SentTotal, JustSent)) return false;
        if (JustSent <= 0) return false;

        SentTotal += JustSent;
    }
    return true;
}

static bool RecvExact(FSocket* S, uint8* Data, int32 Len)
{
    if (!S || !Data || Len < 0) return false;
    if (Len == 0) return true;

    int32 ReadTotal = 0;
    while (ReadTotal < Len)
    {
        if (!WaitReadable(S, kWaitSeconds)) return false;

        int32 JustRead = 0;
        if (!S->Recv(Data + ReadTotal, Len - ReadTotal, JustRead)) return false;
        if (JustRead <= 0) return false;

        ReadTotal += JustRead;
    }
    return true;
}

bool USocketClient::ConnectToServer(const FString& Host, int32 Port)
{
    Close();

    FIPv4Address Addr;
    if (!FIPv4Address::Parse(Host, Addr))
    {
        UE_LOG(LogTemp, Error, TEXT("ConnectToServer: Bad IPv4: %s"), *Host);
        return false;
    }

    TSharedRef<FInternetAddr> InternetAddr =
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();
    InternetAddr->SetIp(Addr.Value);
    InternetAddr->SetPort(Port);

    Socket = FTcpSocketBuilder(TEXT("RLClient"))
        .AsBlocking()
        .WithReceiveBufferSize(2 * 1024 * 1024)
        .WithSendBufferSize(2 * 1024 * 1024);

    if (!Socket)
    {
        UE_LOG(LogTemp, Error, TEXT("ConnectToServer: Failed to create socket"));
        return false;
    }

    Socket->SetNonBlocking(false);

    const bool bOk = Socket->Connect(*InternetAddr);
    UE_LOG(LogTemp, Log, TEXT("SocketClient::Connect %s:%d => %s"), *Host, Port, bOk ? TEXT("OK") : TEXT("FAIL"));

    if (!bOk)
    {
        Close();
        return false;
    }

    return true;
}

bool USocketClient::SendJsonAndReceiveJson(const FString& OutJson, FString& InJson)
{
    InJson.Empty();
    if (!Socket) return false;

    FTCHARToUTF8 Utf8(*OutJson);
    const uint32 PayloadLen = (uint32)Utf8.Length();

    if (PayloadLen == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("SendJsonAndReceiveJson: OutJson is EMPTY"));
        return false;
    }

    const uint32 NetLen = htonl(PayloadLen);

    // 1) Send length + payload
    if (!SendAll(Socket, reinterpret_cast<const uint8*>(&NetLen), 4)) return false;
    if (!SendAll(Socket, reinterpret_cast<const uint8*>(Utf8.Get()), (int32)PayloadLen)) return false;

    // 2) Receive length
    uint32 RecvNetLen = 0;
    if (!RecvExact(Socket, reinterpret_cast<uint8*>(&RecvNetLen), 4)) return false;

    const uint32 RecvLen = ntohl(RecvNetLen);
    if (RecvLen == 0 || RecvLen > (uint32)kMaxPacketBytes)
    {
        UE_LOG(LogTemp, Warning, TEXT("Bad RecvLen: %u"), RecvLen);
        return false;
    }

    // 3) Receive payload EXACT
    TArray<uint8> Buffer;
    Buffer.SetNumUninitialized((int32)RecvLen);

    if (!RecvExact(Socket, Buffer.GetData(), (int32)RecvLen)) return false;

    // 4) UTF8 -> FString using exact length
    const ANSICHAR* AnsiPtr = reinterpret_cast<const ANSICHAR*>(Buffer.GetData());
    FUTF8ToTCHAR Utf8In(AnsiPtr, (int32)RecvLen);
    InJson = FString(Utf8In.Length(), Utf8In.Get());

    // 5) Якщо сервер додає "сміття" після JSON — обрізаємо до останньої '}'
    InJson.TrimStartAndEndInline();

    return true;
}


FString USocketClient::BuildStepJson(const FVector& HiderLoc, const FVector& SeekerLoc, const FVector& Vel, const TArray<float>& HiderSensors, const TArray<float>& SeekerSensors, float Reward, bool bDone, float HiderYaw, float SeekerYaw)
{
    TSharedPtr<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetStringField(TEXT("cmd"), TEXT("step"));

    TSharedPtr<FJsonObject> Obs = MakeShared<FJsonObject>();

    auto Make2 = [](double A, double B)
        {
            TArray<TSharedPtr<FJsonValue>> Arr;
            Arr.Add(MakeShared<FJsonValueNumber>(A));
            Arr.Add(MakeShared<FJsonValueNumber>(B));
            return Arr;
        };

    Obs->SetArrayField(TEXT("hider"), Make2(HiderLoc.X, HiderLoc.Y));
    Obs->SetArrayField(TEXT("seeker"), Make2(SeekerLoc.X, SeekerLoc.Y));
    Obs->SetArrayField(TEXT("vel"), Make2(Vel.X, Vel.Y));

    // Додаємо сенсори Hider'а 
    TArray<TSharedPtr<FJsonValue>> HiderSensorsJson;
    for (float Val : HiderSensors) HiderSensorsJson.Add(MakeShared<FJsonValueNumber>(Val));
    Obs->SetArrayField(TEXT("hider_sensors"), HiderSensorsJson);

    // Додаємо сенсори Seeker'а 
    TArray<TSharedPtr<FJsonValue>> SeekerSensorsJson;
    for (float Val : SeekerSensors) SeekerSensorsJson.Add(MakeShared<FJsonValueNumber>(Val));
    Obs->SetArrayField(TEXT("seeker_sensors"), SeekerSensorsJson);

    // ДОДАЄМО YAW В JSON (Один раз, чисто)
    Obs->SetNumberField(TEXT("hider_yaw"), HiderYaw);
    Obs->SetNumberField(TEXT("seeker_yaw"), SeekerYaw);

    Root->SetObjectField(TEXT("obs"), Obs);

    // ДОДАЄМО НАГОРОДУ ТА СТАТУС ДЛЯ НАВЧАННЯ
    Root->SetNumberField(TEXT("reward"), Reward);
    Root->SetBoolField(TEXT("done"), bDone);

    FString Out;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Out);
    FJsonSerializer::Serialize(Root.ToSharedRef(), Writer);
    return Out;
}

bool USocketClient::ExtractAction(const FString& InJson, float& OutHiderMove, float& OutHiderTurn, float& OutSeekerMove, float& OutSeekerTurn)
{
    // За замовчуванням усі стоять і не крутяться
    OutHiderMove = 0.0f;
    OutHiderTurn = 0.0f;
    OutSeekerMove = 0.0f;
    OutSeekerTurn = 0.0f;

    TSharedPtr<FJsonObject> Obj;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(InJson);

    if (!FJsonSerializer::Deserialize(Reader, Obj) || !Obj.IsValid())
        return false;

    // Дістаємо нові дробові дії для обох ботів
    double TempVal = 0.0;

    if (Obj->TryGetNumberField(TEXT("hider_move"), TempVal)) OutHiderMove = (float)TempVal;
    if (Obj->TryGetNumberField(TEXT("hider_turn"), TempVal)) OutHiderTurn = (float)TempVal;

    if (Obj->TryGetNumberField(TEXT("seeker_move"), TempVal)) OutSeekerMove = (float)TempVal;
    if (Obj->TryGetNumberField(TEXT("seeker_turn"), TempVal)) OutSeekerTurn = (float)TempVal;

    return true;
}

bool USocketClient::Step(const FVector& HiderLoc, const FVector& SeekerLoc, const FVector& Vel, const TArray<float>& HiderSensors, const TArray<float>& SeekerSensors, float Reward, bool bDone, float HiderYaw, float SeekerYaw, float& OutHiderMove, float& OutHiderTurn, float& OutSeekerMove, float& OutSeekerTurn, FString& OutResponseJson)
{
    OutHiderMove = 0.0f;
    OutHiderTurn = 0.0f;
    OutSeekerMove = 0.0f;
    OutSeekerTurn = 0.0f;
    OutResponseJson.Empty();

    // Передаємо нові масиви сенсорів у генератор JSON разом з Reward та bDone
    const FString OutJson = BuildStepJson(HiderLoc, SeekerLoc, Vel, HiderSensors, SeekerSensors, Reward, bDone, HiderYaw, SeekerYaw);

    if (!SendJsonAndReceiveJson(OutJson, OutResponseJson))
        return false;

    return ExtractAction(OutResponseJson, OutHiderMove, OutHiderTurn, OutSeekerMove, OutSeekerTurn);
}

void USocketClient::Close()
{
    if (Socket)
    {
        Socket->Shutdown(ESocketShutdownMode::ReadWrite);
        Socket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Socket);
        Socket = nullptr;
    }
}
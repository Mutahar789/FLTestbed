package org.openmined.syft.networking.requests

import io.reactivex.Single
import org.openmined.syft.networking.datamodels.syft.*
import org.openmined.syft.networking.datamodels.syft.AuthenticationRequest
import org.openmined.syft.networking.datamodels.syft.AuthenticationResponse
import org.openmined.syft.networking.datamodels.syft.CycleRequest
import org.openmined.syft.networking.datamodels.syft.CycleResponseData
import org.openmined.syft.networking.datamodels.syft.ReportRequest
import org.openmined.syft.networking.datamodels.syft.ReportResponseData


internal interface CommunicationAPI {
    fun authenticate(authRequest: AuthenticationRequest): Single<AuthenticationResponse>

    fun getCycle(cycleRequest: CycleRequest): Single<CycleResponseData>

    fun reportModel(reportRequest: ReportRequest): Single<ReportResponseData>

    fun reportStats(reportRequest: ReportStatRequest): Single<ReportStatResponseData>

    fun report(reportRequest: ReportRequest): Single<ReportResponse>

    fun saveFirebaseDeviceToken(tokenRequest: TokenRequest): Single<TokenResponseData>

    fun updateWorkerOnlineStatus(tokenRequest: WorkerStatusRequest): Single<TokenResponseData>

    fun uploadMetrics(metricsRequest: TrainingStatsRequestList): Single<TokenResponseData>
}

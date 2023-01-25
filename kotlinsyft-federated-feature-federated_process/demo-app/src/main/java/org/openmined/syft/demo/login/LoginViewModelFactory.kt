package org.openmined.syft.demo.login

import android.app.Application
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import org.openmined.syft.demo.MyApp
import org.openmined.syft.demo.federated.service.WorkerRepository
import org.openmined.syft.demo.federated.ui.work.WorkInfoViewModel
import org.openmined.syft.domain.SyftConfiguration

@Suppress("UNCHECKED_CAST")
class LoginViewModelFactory(var configuration: SyftConfiguration, var application: MyApp, var baseUrl: String) : ViewModelProvider.Factory {

    override fun <T : ViewModel?> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(LoginViewModel::class.java))
            return LoginViewModel(configuration, application, baseUrl) as T
        throw IllegalArgumentException("unknown view model class")
    }
}
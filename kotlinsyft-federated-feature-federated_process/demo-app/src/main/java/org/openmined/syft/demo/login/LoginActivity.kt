package org.openmined.syft.demo.login

import android.content.Intent
import android.os.Bundle
import android.view.inputmethod.EditorInfo
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.databinding.DataBindingUtil
import androidx.lifecycle.ViewModelProvider
import org.openmined.syft.demo.MyApp
import org.openmined.syft.demo.R
import org.openmined.syft.demo.databinding.ActivityLoginBinding
import org.openmined.syft.demo.federated.ui.main.MnistActivity
import org.openmined.syft.demo.utils.showMessage
import org.openmined.syft.domain.SyftConfiguration
import timber.log.Timber

@ExperimentalUnsignedTypes
@ExperimentalStdlibApi
class LoginActivity : AppCompatActivity() {
    private lateinit var loginViewModel: LoginViewModel
    private lateinit var binding: ActivityLoginBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        supportActionBar?.hide()
        binding = DataBindingUtil.setContentView(this, R.layout.activity_login)
//        loginViewModel = ViewModelProvider(
//            this,
//            ViewModelProvider.NewInstanceFactory()
//        ).get(LoginViewModel::class.java)




        binding.button.setOnClickListener {
            val baseUrl = binding.url.text.toString()
            val config = SyftConfiguration.builder(this, baseUrl)
                    .setMessagingClient(SyftConfiguration.NetworkingClients.HTTP)
                    .setCacheTimeout(0L)
                    .build()
            loginViewModel = initiateViewModel(config, baseUrl)
//            loginViewModel.authenticate {
//                onAuthenticationCompleted(it)
//            }
        }

        binding.url.setOnEditorActionListener { _, actionId, _ ->
            return@setOnEditorActionListener when (actionId) {
                EditorInfo.IME_ACTION_DONE -> {
                    binding.button.performClick()
                    true
                }
                else -> false
            }
        }
    }

    fun onAuthenticationCompleted(throwable: Throwable?) {
        Timber.d("onAuthenticationCompleted")
        throwable?.let {
            throwable.printStackTrace()
            showMessage(getString(R.string.error_in_authentication))
        } ?: startMnistActivity()
    }

    fun startMnistActivity() {
        Timber.d("startMnistActivity")
        val intent = Intent(this, MnistActivity::class.java)
        intent.putExtra("baseURL", loginViewModel.baseUrl)
        intent.putExtra("authToken", loginViewModel.getAuthToken())
        startActivity(intent)
    }

    private fun initiateViewModel(config: SyftConfiguration, baseUrl: String): LoginViewModel {
        return ViewModelProvider(this,LoginViewModelFactory(config, MyApp.getInstance(), baseUrl)).get(LoginViewModel::class.java)
    }
}
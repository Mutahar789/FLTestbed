from txfcm import TXFCMNotification
from twisted.internet import reactor
import abc

class FcmManagerasfdasdafsdasfdasdf(metaclass=abc.ABCMeta):
    def __init__(self):
        self.push_service = TXFCMNotification(api_key="AAAAsdENP3E:APA91bHNWoH_kx72iYfUUIIsDgaUHV_QDqEbWTW3WuXVVH0q-E7VVmoVC-HFnQuMzcSfm6XJxy0X188ofrKpxha0Ky-jsr6CmdyNlzGDpLU5I5FlctGMJ6bBZgm51ydCzoyO0FScDQjB")

    def send_push_old(self):
        "Sending push --==-------------------"
        response_body = {}

        
        registration_ids = ["duuG9hP3SBWw8hNrOuNU1x:APA91bHVolZOYTuH_A_C7zhSAYGpN4IHv1TcHwOObtR_ShRgHgWNr7Vjl2TO8bcOTkrb7X7Mwq4e-tA2_U55kMipPHUiTgkkMJe-nj6jjARUhpM1icRu5KuFyBaxB3EzesP-5pDgLLoC"]
        
        df = self.push_service.notify_multiple_devices(registration_ids=registration_ids, message_title="This is message title", message_body="this is message_body")

        #def got_result(result):
        #    print(result)

        #df.addBoth(got_result)
        reactor.run()

        response_body["RESPONSE_MSG.ERROR"] = "str(e)"

        return Response(json.dumps(response_body), status=200, mimetype="application/json")
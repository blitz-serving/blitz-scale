# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: generate.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0egenerate.proto\x12\x0bgenerate.v2\"\x0f\n\rHealthRequest\"\x1f\n\x0eHealthResponse\x12\r\n\x05state\x18\x01 \x02(\t\"4\n\x0cRelayRequest\x12\x0c\n\x04rank\x18\x01 \x02(\x05\x12\x16\n\x0erelax_not_head\x18\x02 \x02(\x08\"2\n\rRelayResponse\x12\x10\n\x08\x62\x61tch_id\x18\x01 \x01(\x04\x12\x0f\n\x07seq_num\x18\x02 \x01(\r\"8\n\x10\x42roadcastRequest\x12\x11\n\tsrc_ranks\x18\x01 \x03(\x05\x12\x11\n\tdst_ranks\x18\x02 \x03(\x05\"\x13\n\x11\x42roadcastResponse\"\x14\n\x12ResetStatusRequest\"\x15\n\x13ResetStatusResponse\"\x17\n\x15SetStatusReadyRequest\"\x18\n\x16SetStatusReadyResponse\"\r\n\x0bInfoRequest\"t\n\x0cInfoResponse\x12\x18\n\x10requires_padding\x18\x01 \x02(\x08\x12\r\n\x05\x64type\x18\x02 \x02(\t\x12\x13\n\x0b\x64\x65vice_type\x18\x03 \x02(\t\x12\x13\n\x0bwindow_size\x18\x04 \x01(\r\x12\x11\n\tspeculate\x18\x05 \x02(\r\"\x19\n\x17ServiceDiscoveryRequest\"D\n\x18ServiceDiscoveryResponse\x12\x0c\n\x04urls\x18\x01 \x03(\t\x12\x1a\n\x12ranks_view_in_json\x18\x02 \x02(\t\"\x1f\n\x11\x43learCacheRequest\x12\n\n\x02id\x18\x01 \x01(\x04\"\x14\n\x12\x43learCacheResponse\"\xb2\x01\n\x1aNextTokenChooserParameters\x12\x13\n\x0btemperature\x18\x01 \x02(\x02\x12\r\n\x05top_k\x18\x02 \x02(\r\x12\r\n\x05top_p\x18\x03 \x02(\x02\x12\x11\n\ttypical_p\x18\x04 \x02(\x02\x12\x11\n\tdo_sample\x18\x05 \x02(\x08\x12\x0c\n\x04seed\x18\x06 \x02(\x04\x12\x1a\n\x12repetition_penalty\x18\x07 \x02(\x02\x12\x11\n\twatermark\x18\x08 \x02(\x08\"f\n\x1aStoppingCriteriaParameters\x12\x16\n\x0emax_new_tokens\x18\x01 \x02(\r\x12\x16\n\x0estop_sequences\x18\x02 \x03(\t\x12\x18\n\x10ignore_eos_token\x18\x03 \x02(\x08\"\x80\x02\n\x07Request\x12\n\n\x02id\x18\x01 \x02(\x04\x12\x0e\n\x06inputs\x18\x02 \x02(\t\x12\x10\n\x08truncate\x18\x03 \x01(\r\x12;\n\nparameters\x18\x04 \x01(\x0b\x32\'.generate.v2.NextTokenChooserParameters\x12\x44\n\x13stopping_parameters\x18\x05 \x02(\x0b\x32\'.generate.v2.StoppingCriteriaParameters\x12\x18\n\x10prefill_logprobs\x18\x06 \x02(\x08\x12\x14\n\x0ctop_n_tokens\x18\x07 \x02(\r\x12\x14\n\x0cinput_tokens\x18\x08 \x03(\r\"]\n\x05\x42\x61tch\x12\n\n\x02id\x18\x01 \x02(\x04\x12&\n\x08requests\x18\x02 \x03(\x0b\x32\x14.generate.v2.Request\x12\x0c\n\x04size\x18\x03 \x02(\r\x12\x12\n\nmax_tokens\x18\x04 \x02(\r\"P\n\x0b\x43\x61\x63hedBatch\x12\n\n\x02id\x18\x01 \x02(\x04\x12\x13\n\x0brequest_ids\x18\x02 \x03(\x04\x12\x0c\n\x04size\x18\x03 \x02(\r\x12\x12\n\nmax_tokens\x18\x04 \x02(\r\"w\n\rGeneratedText\x12\x0c\n\x04text\x18\x01 \x02(\t\x12\x18\n\x10generated_tokens\x18\x02 \x02(\r\x12\x30\n\rfinish_reason\x18\x03 \x02(\x0e\x32\x19.generate.v2.FinishReason\x12\x0c\n\x04seed\x18\x04 \x01(\x04\"J\n\x06Tokens\x12\x0b\n\x03ids\x18\x01 \x03(\r\x12\x10\n\x08logprobs\x18\x02 \x03(\x02\x12\r\n\x05texts\x18\x03 \x03(\t\x12\x12\n\nis_special\x18\x04 \x03(\x08\"\xcf\x01\n\nGeneration\x12\x12\n\nrequest_id\x18\x01 \x02(\x04\x12+\n\x0eprefill_tokens\x18\x02 \x01(\x0b\x32\x13.generate.v2.Tokens\x12#\n\x06tokens\x18\x03 \x02(\x0b\x32\x13.generate.v2.Tokens\x12\x32\n\x0egenerated_text\x18\x04 \x01(\x0b\x32\x1a.generate.v2.GeneratedText\x12\'\n\ntop_tokens\x18\x05 \x03(\x0b\x32\x13.generate.v2.Tokens\";\n\x12\x46ilterBatchRequest\x12\x10\n\x08\x62\x61tch_id\x18\x01 \x02(\x04\x12\x13\n\x0brequest_ids\x18\x02 \x03(\x04\">\n\x13\x46ilterBatchResponse\x12\'\n\x05\x62\x61tch\x18\x01 \x02(\x0b\x32\x18.generate.v2.CachedBatch\"3\n\x0ePrefillRequest\x12!\n\x05\x62\x61tch\x18\x01 \x02(\x0b\x32\x12.generate.v2.Batch\"\xa1\x01\n\x0fPrefillResponse\x12,\n\x0bgenerations\x18\x01 \x03(\x0b\x32\x17.generate.v2.Generation\x12\'\n\x05\x62\x61tch\x18\x02 \x01(\x0b\x32\x18.generate.v2.CachedBatch\x12\x12\n\nforward_ns\x18\x03 \x02(\x04\x12\x11\n\tdecode_ns\x18\x04 \x02(\x04\x12\x10\n\x08total_ns\x18\x05 \x02(\x04\":\n\rDecodeRequest\x12)\n\x07\x62\x61tches\x18\x01 \x03(\x0b\x32\x18.generate.v2.CachedBatch\"\xb3\x01\n\x0e\x44\x65\x63odeResponse\x12,\n\x0bgenerations\x18\x01 \x03(\x0b\x32\x17.generate.v2.Generation\x12\'\n\x05\x62\x61tch\x18\x02 \x01(\x0b\x32\x18.generate.v2.CachedBatch\x12\x12\n\nforward_ns\x18\x03 \x02(\x04\x12\x11\n\tdecode_ns\x18\x04 \x02(\x04\x12\x10\n\x08total_ns\x18\x05 \x02(\x04\x12\x11\n\tconcat_ns\x18\x06 \x01(\x04\"\x82\x01\n\rWarmupRequest\x12!\n\x05\x62\x61tch\x18\x01 \x01(\x0b\x32\x12.generate.v2.Batch\x12\x18\n\x10max_input_length\x18\x02 \x01(\r\x12\x1a\n\x12max_prefill_tokens\x18\x03 \x01(\r\x12\x18\n\x10max_total_tokens\x18\x04 \x01(\r\"4\n\x0eWarmupResponse\x12\"\n\x1amax_supported_total_tokens\x18\x01 \x01(\r\" \n\x11SendParamsRequest\x12\x0b\n\x03\x64st\x18\x01 \x02(\r\"\x14\n\x12SendParamsResponse\" \n\x11RecvParamsRequest\x12\x0b\n\x03src\x18\x01 \x02(\r\"\x14\n\x12RecvParamsResponse\"j\n\x11LoadParamsRequest\x12-\n\tload_case\x18\x01 \x02(\x0e\x32\x1a.generate.v2.LoadParamCase\x12\x12\n\nmodel_name\x18\x02 \x02(\t\x12\x12\n\nmodel_path\x18\x03 \x01(\t\"\x14\n\x12LoadParamsResponse\"|\n\x0cPipeParaInfo\x12\x19\n\x0f\x65mbedding_layer\x18\x01 \x01(\rH\x00\x12\x11\n\x07lm_head\x18\x02 \x01(\rH\x00\x12\x13\n\ttfm_layer\x18\x03 \x01(\rH\x00\x12\x1a\n\x12num_layer_per_rank\x18\x04 \x03(\rB\r\n\x0bstart_layer\"\x8a\x01\n\x10PrefillV2Request\x12!\n\x05\x62\x61tch\x18\x01 \x02(\x0b\x32\x12.generate.v2.Batch\x12\x14\n\x0c\x66orward_case\x18\x02 \x02(\r\x12*\n\x07pp_info\x18\x03 \x01(\x0b\x32\x19.generate.v2.PipeParaInfo\x12\x11\n\tpipe_peer\x18\x04 \x01(\r\"\xcf\x01\n\x11PrefillV2Response\x12,\n\x0bgenerations\x18\x01 \x03(\x0b\x32\x17.generate.v2.Generation\x12\'\n\x05\x62\x61tch\x18\x02 \x01(\x0b\x32\x18.generate.v2.CachedBatch\x12*\n\x07pp_info\x18\x03 \x01(\x0b\x32\x19.generate.v2.PipeParaInfo\x12\x12\n\nforward_ns\x18\x04 \x02(\x04\x12\x11\n\tdecode_ns\x18\x05 \x02(\x04\x12\x10\n\x08total_ns\x18\x06 \x02(\x04\"\xa1\x01\n\x11ZagPrefillRequest\x12!\n\x05\x62\x61tch\x18\x01 \x02(\x0b\x32\x12.generate.v2.Batch\x12\x14\n\x0c\x66orward_case\x18\x02 \x02(\r\x12*\n\x07pp_info\x18\x03 \x01(\x0b\x32\x19.generate.v2.PipeParaInfo\x12\x12\n\nzag_layers\x18\x04 \x02(\r\x12\x13\n\x0bzag_seq_num\x18\x05 \x02(\r\"k\n\x0f\x44\x65\x63odeV2Request\x12)\n\x07\x62\x61tches\x18\x01 \x03(\x0b\x32\x18.generate.v2.CachedBatch\x12-\n\x10last_iter_tokens\x18\x02 \x03(\x0b\x32\x13.generate.v2.Tokens\"{\n\x10\x44\x65\x63odeV2Response\x12,\n\x0bgenerations\x18\x01 \x03(\x0b\x32\x17.generate.v2.Generation\x12\'\n\x05\x62\x61tch\x18\x02 \x01(\x0b\x32\x18.generate.v2.CachedBatch\x12\x10\n\x08total_ns\x18\x03 \x02(\x04\"M\n\x0eMigrateRequest\x12!\n\x05\x62\x61tch\x18\x01 \x02(\x0b\x32\x12.generate.v2.Batch\x12\x0b\n\x03src\x18\x02 \x03(\r\x12\x0b\n\x03\x64st\x18\x03 \x03(\r\"\x95\x01\n\x15MigratePartialRequest\x12!\n\x05\x62\x61tch\x18\x01 \x02(\x0b\x32\x12.generate.v2.Batch\x12,\n\nfst_or_snd\x18\x02 \x02(\x0e\x32\x18.generate.v2.PartialCase\x12\x11\n\tnum_layer\x18\x03 \x02(\r\x12\x0b\n\x03src\x18\x04 \x02(\r\x12\x0b\n\x03\x64st\x18\x05 \x02(\r\":\n\x0fMigrateResponse\x12\'\n\x05\x62\x61tch\x18\x01 \x02(\x0b\x32\x18.generate.v2.CachedBatch\"O\n\x10ImmigrateRequest\x12!\n\x05\x62\x61tch\x18\x01 \x02(\x0b\x32\x12.generate.v2.Batch\x12\x0b\n\x03src\x18\x02 \x03(\r\x12\x0b\n\x03\x64st\x18\x03 \x03(\r\"\x97\x01\n\x17ImmigratePartialRequest\x12!\n\x05\x62\x61tch\x18\x01 \x02(\x0b\x32\x12.generate.v2.Batch\x12,\n\nfst_or_snd\x18\x02 \x02(\x0e\x32\x18.generate.v2.PartialCase\x12\x11\n\tnum_layer\x18\x03 \x02(\r\x12\x0b\n\x03src\x18\x04 \x02(\r\x12\x0b\n\x03\x64st\x18\x05 \x02(\r\"<\n\x11ImmigrateResponse\x12\'\n\x05\x62\x61tch\x18\x01 \x02(\x0b\x32\x18.generate.v2.CachedBatch\"\x15\n\x13WaitRdmaDoneRequest\"\x16\n\x14WaitRdmaDoneResponse*f\n\x0c\x46inishReason\x12\x18\n\x14\x46INISH_REASON_LENGTH\x10\x00\x12\x1b\n\x17\x46INISH_REASON_EOS_TOKEN\x10\x01\x12\x1f\n\x1b\x46INISH_REASON_STOP_SEQUENCE\x10\x02*;\n\rLoadParamCase\x12\x16\n\x12LOAD_FROM_HOST_MEM\x10\x00\x12\x12\n\x0eLOAD_FROM_DISK\x10\x01*\x9d\x01\n\x0bPrefillCase\x12\x17\n\x13PREFILL_CASE_NORMAL\x10\x00\x12\x18\n\x14PREFILL_CASE_MIGRATE\x10\x01\x12$\n PREFILL_CASE_MIGRATE_PROGRESSIVE\x10\x02\x12\x19\n\x15PREFILL_CASE_NAIVE_PP\x10\x04\x12\x1a\n\x16PREFILL_CASE_IMMIGRATE\x10\x08*>\n\x0bPartialCase\x12\x16\n\x12PARTIAL_CASE_FIRST\x10\x01\x12\x17\n\x13PARTIAL_CASE_SECOND\x10\x02\x32\xaa\x0f\n\x15TextGenerationService\x12=\n\x04Info\x12\x18.generate.v2.InfoRequest\x1a\x19.generate.v2.InfoResponse\"\x00\x12\x61\n\x10ServiceDiscovery\x12$.generate.v2.ServiceDiscoveryRequest\x1a%.generate.v2.ServiceDiscoveryResponse\"\x00\x12M\n\nClearCache\x12\x1e.generate.v2.ClearCacheRequest\x1a\x1f.generate.v2.ClearCacheResponse\x12P\n\x0b\x46ilterBatch\x12\x1f.generate.v2.FilterBatchRequest\x1a .generate.v2.FilterBatchResponse\x12\x41\n\x06Warmup\x12\x1a.generate.v2.WarmupRequest\x1a\x1b.generate.v2.WarmupResponse\x12\x44\n\x07Prefill\x12\x1b.generate.v2.PrefillRequest\x1a\x1c.generate.v2.PrefillResponse\x12\x41\n\x06\x44\x65\x63ode\x12\x1a.generate.v2.DecodeRequest\x1a\x1b.generate.v2.DecodeResponse\x12\x41\n\x06Health\x12\x1a.generate.v2.HealthRequest\x1a\x1b.generate.v2.HealthResponse\x12M\n\nSendParams\x12\x1e.generate.v2.SendParamsRequest\x1a\x1f.generate.v2.SendParamsResponse\x12M\n\nRecvParams\x12\x1e.generate.v2.RecvParamsRequest\x1a\x1f.generate.v2.RecvParamsResponse\x12M\n\nLoadParams\x12\x1e.generate.v2.LoadParamsRequest\x1a\x1f.generate.v2.LoadParamsResponse\x12J\n\tPrefillV2\x12\x1d.generate.v2.PrefillV2Request\x1a\x1e.generate.v2.PrefillV2Response\x12G\n\x08\x44\x65\x63odeV2\x12\x1c.generate.v2.DecodeV2Request\x1a\x1d.generate.v2.DecodeV2Response\x12L\n\nZagPrefill\x12\x1e.generate.v2.ZagPrefillRequest\x1a\x1e.generate.v2.PrefillV2Response\x12\x44\n\x07Migrate\x12\x1b.generate.v2.MigrateRequest\x1a\x1c.generate.v2.MigrateResponse\x12J\n\tImmigrate\x12\x1d.generate.v2.ImmigrateRequest\x1a\x1e.generate.v2.ImmigrateResponse\x12R\n\x0eMigratePartial\x12\".generate.v2.MigratePartialRequest\x1a\x1c.generate.v2.MigrateResponse\x12X\n\x10ImmigratePartial\x12$.generate.v2.ImmigratePartialRequest\x1a\x1e.generate.v2.ImmigrateResponse\x12S\n\x0cWaitRdmaDone\x12 .generate.v2.WaitRdmaDoneRequest\x1a!.generate.v2.WaitRdmaDoneResponse\x12P\n\x0bResetStatus\x12\x1f.generate.v2.ResetStatusRequest\x1a .generate.v2.ResetStatusResponse\x12Y\n\x0eSetStatusReady\x12\".generate.v2.SetStatusReadyRequest\x1a#.generate.v2.SetStatusReadyResponse\x12>\n\x05Relay\x12\x19.generate.v2.RelayRequest\x1a\x1a.generate.v2.RelayResponse\x12M\n\x0cNvlBroadcast\x12\x1d.generate.v2.BroadcastRequest\x1a\x1e.generate.v2.BroadcastResponse\x12N\n\rRdmaBroadcast\x12\x1d.generate.v2.BroadcastRequest\x1a\x1e.generate.v2.BroadcastResponse\x12N\n\rTanzBroadcast\x12\x1d.generate.v2.BroadcastRequest\x1a\x1e.generate.v2.BroadcastResponse')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'generate_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_FINISHREASON']._serialized_start=4298
  _globals['_FINISHREASON']._serialized_end=4400
  _globals['_LOADPARAMCASE']._serialized_start=4402
  _globals['_LOADPARAMCASE']._serialized_end=4461
  _globals['_PREFILLCASE']._serialized_start=4464
  _globals['_PREFILLCASE']._serialized_end=4621
  _globals['_PARTIALCASE']._serialized_start=4623
  _globals['_PARTIALCASE']._serialized_end=4685
  _globals['_HEALTHREQUEST']._serialized_start=31
  _globals['_HEALTHREQUEST']._serialized_end=46
  _globals['_HEALTHRESPONSE']._serialized_start=48
  _globals['_HEALTHRESPONSE']._serialized_end=79
  _globals['_RELAYREQUEST']._serialized_start=81
  _globals['_RELAYREQUEST']._serialized_end=133
  _globals['_RELAYRESPONSE']._serialized_start=135
  _globals['_RELAYRESPONSE']._serialized_end=185
  _globals['_BROADCASTREQUEST']._serialized_start=187
  _globals['_BROADCASTREQUEST']._serialized_end=243
  _globals['_BROADCASTRESPONSE']._serialized_start=245
  _globals['_BROADCASTRESPONSE']._serialized_end=264
  _globals['_RESETSTATUSREQUEST']._serialized_start=266
  _globals['_RESETSTATUSREQUEST']._serialized_end=286
  _globals['_RESETSTATUSRESPONSE']._serialized_start=288
  _globals['_RESETSTATUSRESPONSE']._serialized_end=309
  _globals['_SETSTATUSREADYREQUEST']._serialized_start=311
  _globals['_SETSTATUSREADYREQUEST']._serialized_end=334
  _globals['_SETSTATUSREADYRESPONSE']._serialized_start=336
  _globals['_SETSTATUSREADYRESPONSE']._serialized_end=360
  _globals['_INFOREQUEST']._serialized_start=362
  _globals['_INFOREQUEST']._serialized_end=375
  _globals['_INFORESPONSE']._serialized_start=377
  _globals['_INFORESPONSE']._serialized_end=493
  _globals['_SERVICEDISCOVERYREQUEST']._serialized_start=495
  _globals['_SERVICEDISCOVERYREQUEST']._serialized_end=520
  _globals['_SERVICEDISCOVERYRESPONSE']._serialized_start=522
  _globals['_SERVICEDISCOVERYRESPONSE']._serialized_end=590
  _globals['_CLEARCACHEREQUEST']._serialized_start=592
  _globals['_CLEARCACHEREQUEST']._serialized_end=623
  _globals['_CLEARCACHERESPONSE']._serialized_start=625
  _globals['_CLEARCACHERESPONSE']._serialized_end=645
  _globals['_NEXTTOKENCHOOSERPARAMETERS']._serialized_start=648
  _globals['_NEXTTOKENCHOOSERPARAMETERS']._serialized_end=826
  _globals['_STOPPINGCRITERIAPARAMETERS']._serialized_start=828
  _globals['_STOPPINGCRITERIAPARAMETERS']._serialized_end=930
  _globals['_REQUEST']._serialized_start=933
  _globals['_REQUEST']._serialized_end=1189
  _globals['_BATCH']._serialized_start=1191
  _globals['_BATCH']._serialized_end=1284
  _globals['_CACHEDBATCH']._serialized_start=1286
  _globals['_CACHEDBATCH']._serialized_end=1366
  _globals['_GENERATEDTEXT']._serialized_start=1368
  _globals['_GENERATEDTEXT']._serialized_end=1487
  _globals['_TOKENS']._serialized_start=1489
  _globals['_TOKENS']._serialized_end=1563
  _globals['_GENERATION']._serialized_start=1566
  _globals['_GENERATION']._serialized_end=1773
  _globals['_FILTERBATCHREQUEST']._serialized_start=1775
  _globals['_FILTERBATCHREQUEST']._serialized_end=1834
  _globals['_FILTERBATCHRESPONSE']._serialized_start=1836
  _globals['_FILTERBATCHRESPONSE']._serialized_end=1898
  _globals['_PREFILLREQUEST']._serialized_start=1900
  _globals['_PREFILLREQUEST']._serialized_end=1951
  _globals['_PREFILLRESPONSE']._serialized_start=1954
  _globals['_PREFILLRESPONSE']._serialized_end=2115
  _globals['_DECODEREQUEST']._serialized_start=2117
  _globals['_DECODEREQUEST']._serialized_end=2175
  _globals['_DECODERESPONSE']._serialized_start=2178
  _globals['_DECODERESPONSE']._serialized_end=2357
  _globals['_WARMUPREQUEST']._serialized_start=2360
  _globals['_WARMUPREQUEST']._serialized_end=2490
  _globals['_WARMUPRESPONSE']._serialized_start=2492
  _globals['_WARMUPRESPONSE']._serialized_end=2544
  _globals['_SENDPARAMSREQUEST']._serialized_start=2546
  _globals['_SENDPARAMSREQUEST']._serialized_end=2578
  _globals['_SENDPARAMSRESPONSE']._serialized_start=2580
  _globals['_SENDPARAMSRESPONSE']._serialized_end=2600
  _globals['_RECVPARAMSREQUEST']._serialized_start=2602
  _globals['_RECVPARAMSREQUEST']._serialized_end=2634
  _globals['_RECVPARAMSRESPONSE']._serialized_start=2636
  _globals['_RECVPARAMSRESPONSE']._serialized_end=2656
  _globals['_LOADPARAMSREQUEST']._serialized_start=2658
  _globals['_LOADPARAMSREQUEST']._serialized_end=2764
  _globals['_LOADPARAMSRESPONSE']._serialized_start=2766
  _globals['_LOADPARAMSRESPONSE']._serialized_end=2786
  _globals['_PIPEPARAINFO']._serialized_start=2788
  _globals['_PIPEPARAINFO']._serialized_end=2912
  _globals['_PREFILLV2REQUEST']._serialized_start=2915
  _globals['_PREFILLV2REQUEST']._serialized_end=3053
  _globals['_PREFILLV2RESPONSE']._serialized_start=3056
  _globals['_PREFILLV2RESPONSE']._serialized_end=3263
  _globals['_ZAGPREFILLREQUEST']._serialized_start=3266
  _globals['_ZAGPREFILLREQUEST']._serialized_end=3427
  _globals['_DECODEV2REQUEST']._serialized_start=3429
  _globals['_DECODEV2REQUEST']._serialized_end=3536
  _globals['_DECODEV2RESPONSE']._serialized_start=3538
  _globals['_DECODEV2RESPONSE']._serialized_end=3661
  _globals['_MIGRATEREQUEST']._serialized_start=3663
  _globals['_MIGRATEREQUEST']._serialized_end=3740
  _globals['_MIGRATEPARTIALREQUEST']._serialized_start=3743
  _globals['_MIGRATEPARTIALREQUEST']._serialized_end=3892
  _globals['_MIGRATERESPONSE']._serialized_start=3894
  _globals['_MIGRATERESPONSE']._serialized_end=3952
  _globals['_IMMIGRATEREQUEST']._serialized_start=3954
  _globals['_IMMIGRATEREQUEST']._serialized_end=4033
  _globals['_IMMIGRATEPARTIALREQUEST']._serialized_start=4036
  _globals['_IMMIGRATEPARTIALREQUEST']._serialized_end=4187
  _globals['_IMMIGRATERESPONSE']._serialized_start=4189
  _globals['_IMMIGRATERESPONSE']._serialized_end=4249
  _globals['_WAITRDMADONEREQUEST']._serialized_start=4251
  _globals['_WAITRDMADONEREQUEST']._serialized_end=4272
  _globals['_WAITRDMADONERESPONSE']._serialized_start=4274
  _globals['_WAITRDMADONERESPONSE']._serialized_end=4296
  _globals['_TEXTGENERATIONSERVICE']._serialized_start=4688
  _globals['_TEXTGENERATIONSERVICE']._serialized_end=6650
# @@protoc_insertion_point(module_scope)

from fastapi import APIRouter, HTTPException
from ..config import settings


router = APIRouter(prefix=f"{settings.API_PREFIX}", tags=["placeholders"])


def not_implemented(name: str):
    raise HTTPException(status_code=501, detail=f"{name} not implemented yet")


@router.get("/board/concept/list")
async def board_concept_list():
    not_implemented("stock_board_concept_name_em")


@router.get("/board/concept/constituents")
async def board_concept_cons():
    not_implemented("stock_board_concept_cons_em")


@router.get("/board/concept/spot")
async def board_concept_spot():
    not_implemented("stock_board_concept_spot_em")


@router.get("/board/industry/list")
async def board_industry_list():
    not_implemented("stock_board_industry_name_em")


@router.get("/board/industry/constituents")
async def board_industry_cons():
    not_implemented("stock_board_industry_cons_em")


@router.get("/board/industry/spot")
async def board_industry_spot():
    not_implemented("stock_board_industry_spot_em")


@router.get("/hsgt/fundflow")
async def hsgt_fundflow():
    not_implemented("stock_hsgt_fundflow")


@router.get("/hsgt/hold")
async def hsgt_hold():
    not_implemented("stock_hsgt_hold_stock")


@router.get("/dividend/em")
async def dividend_em():
    not_implemented("stock_dividend_em")


@router.get("/dividend/ths")
async def dividend_ths():
    not_implemented("stock_dividend_ths")


@router.get("/disclosure")
async def disclosure():
    not_implemented("stock_info_disclosure / yjbb / yjyg")


@router.get("/fundflow/ths")
async def fundflow_ths():
    not_implemented("stock_individual_fund_flow_ths, stock_market_fund_flow_ths")


@router.get("/fundflow/em")
async def fundflow_em():
    not_implemented("stock_individual_fund_flow_em, stock_market_fund_flow_em")


@router.get("/valuations/pe")
async def valuations_pe():
    not_implemented("stock_a_all_pe")


@router.get("/valuations/pb_equal_weight")
async def valuations_pb_equal():
    not_implemented("stock_a_equal_weight_pb")


@router.get("/valuations/median")
async def valuations_median():
    not_implemented("stock_a_median_pe_pb")


@router.get("/buffett")
async def buffett_index():
    not_implemented("stock_buffett_index")


@router.get("/risk_premium")
async def risk_premium():
    not_implemented("stock_risk_premium")


@router.get("/margin/em")
async def margin_em():
    not_implemented("stock_margin_em")


@router.get("/esg/rating")
async def esg_rating():
    not_implemented("stock_esg_rating_em")


@router.get("/esg/full")
async def esg_full():
    not_implemented("stock_esg_full_em")


@router.get("/esg/green_bonds")
async def esg_green():
    not_implemented("stock_green_bond_em")





